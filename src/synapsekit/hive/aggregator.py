"""Self-hostable Hive aggregation core and SQLite persistence."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from ..audit.signer import verify_signature
from .types import (
    HIVE_SCHEMA_VERSION,
    ContributionEnvelope,
    ContributionPayload,
    ShareScope,
    Suggestion,
    TransparencyReport,
    b64decode,
)


class HiveAggregatorError(ValueError):
    """Raised when an envelope cannot be accepted or queried safely."""


class HiveStore(Protocol):
    def put(self, envelope: ContributionEnvelope) -> bool: ...

    def list(
        self, *, scope_id: str, include_revoked: bool = False
    ) -> list[ContributionEnvelope]: ...

    def revoke(self, *, contributor_id: str, scope_id: str) -> int: ...


class SQLiteHiveStore:
    """Small, restart-safe reference store using only the Python stdlib."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        # ``check_same_thread=False`` lets the single connection be used from the
        # worker threads that ``asyncio.to_thread`` dispatches store calls onto,
        # but sqlite3 is not internally thread-safe for concurrent use of one
        # connection, so every access is serialized through this lock.
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._ensure_schema()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _ensure_schema(self) -> None:
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS hive_contributions (
                    contribution_id TEXT PRIMARY KEY,
                    schema_version TEXT NOT NULL,
                    scope_id TEXT NOT NULL,
                    contributor_id TEXT NOT NULL,
                    envelope_json TEXT NOT NULL,
                    received_at REAL NOT NULL,
                    revoked INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS hive_scope_idx
                  ON hive_contributions(scope_id, revoked, received_at);
                CREATE INDEX IF NOT EXISTS hive_contributor_idx
                  ON hive_contributions(scope_id, contributor_id);
                """
            )
            self._connection.commit()

    def put(self, envelope: ContributionEnvelope) -> bool:
        payload = envelope.payload
        if payload is None:
            raise HiveAggregatorError("SQLiteHiveStore requires a decrypted envelope")
        with self._lock:
            cursor = self._connection.execute(
                """
                INSERT OR IGNORE INTO hive_contributions
                (contribution_id, schema_version, scope_id, contributor_id, envelope_json, received_at, revoked)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope.contribution_id,
                    HIVE_SCHEMA_VERSION,
                    payload.scope_id,
                    envelope.contributor_id,
                    json.dumps(envelope.to_dict(), sort_keys=True),
                    envelope.received_at,
                    int(envelope.revoked),
                ),
            )
            self._connection.commit()
            return cursor.rowcount == 1

    def list(self, *, scope_id: str, include_revoked: bool = False) -> list[ContributionEnvelope]:
        query = "SELECT envelope_json, revoked FROM hive_contributions WHERE scope_id = ?"
        params: list[object] = [scope_id]
        if not include_revoked:
            query += " AND revoked = 0"
        query += " ORDER BY received_at ASC"
        with self._lock:
            rows = self._connection.execute(query, params).fetchall()
        envelopes: list[ContributionEnvelope] = []
        for row in rows:
            envelope = ContributionEnvelope.from_dict(json.loads(row["envelope_json"]))
            # The revoked column is the source of truth (revoke() updates it,
            # not the stored JSON blob), so overlay it onto the envelope.
            revoked = bool(row["revoked"])
            if revoked != envelope.revoked:
                envelope = replace(envelope, revoked=revoked)
            envelopes.append(envelope)
        return envelopes

    def revoke(self, *, contributor_id: str, scope_id: str) -> int:
        with self._lock:
            cursor = self._connection.execute(
                "UPDATE hive_contributions SET revoked = 1 WHERE contributor_id = ? AND scope_id = ?",
                (contributor_id, scope_id),
            )
            self._connection.commit()
            return cursor.rowcount


class HiveAuthorizer(Protocol):
    def authorize(self, *, scope_id: str, actor: str | None, operation: str) -> bool: ...


class AllowAllAuthorizer:
    """Explicitly local-only authorization suitable for an in-process client."""

    def authorize(self, *, scope_id: str, actor: str | None, operation: str) -> bool:
        return True


class HiveAggregator:
    """Validate, persist, and aggregate privacy-processed contributions."""

    def __init__(
        self,
        store: HiveStore | None = None,
        *,
        encryption_key: bytes | None = None,
        trusted_keys: Mapping[str, bytes] | None = None,
        authorizer: HiveAuthorizer | None = None,
        clock: Callable[[], float] = time.time,
        max_clock_skew: float = 900.0,
    ) -> None:
        self.store = store or SQLiteHiveStore()
        self.encryption_key = encryption_key
        self.trusted_keys = dict(trusted_keys or {})
        self.authorizer = authorizer or AllowAllAuthorizer()
        self.clock = clock
        self.max_clock_skew = max_clock_skew

    def submit(self, envelope: ContributionEnvelope, *, actor: str | None = None) -> str:
        payload = self._validate_and_decrypt(envelope)
        if not self.authorizer.authorize(
            scope_id=payload.scope_id, actor=actor, operation="contribute"
        ):
            raise HiveAggregatorError("contribution is not authorized for this scope")
        now = self.clock()
        if abs(now - payload.generated_at) > self.max_clock_skew:
            raise HiveAggregatorError("contribution timestamp is outside the accepted clock window")
        if not payload.patterns:
            raise HiveAggregatorError("contribution contains no safe pattern observations")
        accepted = self.store.put(
            ContributionEnvelope(
                payload=payload,
                contributor_id=envelope.contributor_id,
                public_key=envelope.public_key,
                signature=envelope.signature,
                key_id=envelope.key_id,
                received_at=envelope.received_at,
            )
        )
        if not accepted:
            raise HiveAggregatorError("duplicate contribution")
        return str(envelope.contribution_id)

    def suggestions(
        self,
        *,
        scope_id: str,
        query: str | None = None,
        minimum_cohort: int = 3,
        limit: int = 20,
    ) -> list[Suggestion]:
        if minimum_cohort < 1 or limit < 1:
            raise ValueError("minimum_cohort and limit must be positive")
        envelopes = self.store.list(scope_id=scope_id)
        contributors = {item.contributor_id for item in envelopes}
        if len(contributors) < minimum_cohort:
            return []
        observations: dict[str, list[float]] = defaultdict(list)
        pattern_contributors: dict[str, set[str]] = defaultdict(set)
        categories: dict[str, str] = {}
        for envelope in envelopes:
            assert envelope.payload is not None
            for pattern in envelope.payload.patterns:
                observations[pattern.key].append(pattern.value)
                pattern_contributors[pattern.key].add(envelope.contributor_id)
                categories[pattern.key] = pattern.category
        query_terms = {term.lower() for term in (query or "").split() if term}
        ranked: list[Suggestion] = []
        for key, values in observations.items():
            if query_terms and not query_terms.intersection({key.lower(), *key.lower().split(":")}):
                continue
            # Distinct contributors exhibiting the pattern, not raw observation
            # count — a single contributor may submit several envelopes per day.
            contributor_count = len(pattern_contributors[key])
            prevalence = min(
                1.0, max(0.0, sum(min(1.0, max(0.0, v)) for v in values) / len(contributors))
            )
            confidence = min(1.0, prevalence * (contributor_count / len(contributors)))
            category, short_key = key.split(":", 1)
            ranked.append(
                Suggestion(
                    key=key,
                    category=category,
                    statement=f"{prevalence:.0%} of similar projects show the {short_key} pattern.",
                    prevalence=prevalence,
                    confidence=confidence,
                    contributor_count=contributor_count,
                    cohort_size=len(contributors),
                    scope=ShareScope.COMMUNITY if scope_id == "community" else ShareScope.TEAM,
                )
            )
        ranked.sort(key=lambda item: (-item.confidence, -item.prevalence, item.key))
        return ranked[:limit]

    def transparency(self, *, contributor_id: str, scope_id: str) -> TransparencyReport:
        envelopes = self.store.list(scope_id=scope_id, include_revoked=True)
        mine = [item for item in envelopes if item.contributor_id == contributor_id]
        selected: set[str] = set()
        epsilon = 0.0
        revoked = bool(mine) and all(item.revoked for item in mine)
        for envelope in mine:
            if envelope.payload is None:
                continue
            selected.update(pattern.key for pattern in envelope.payload.patterns)
            epsilon += envelope.payload.epsilon
        scope = (
            ShareScope.LOCAL
            if scope_id == "local"
            else ShareScope.COMMUNITY
            if scope_id == "community"
            else ShareScope.TEAM
        )
        return TransparencyReport(
            scope=scope,
            scope_id=scope_id,
            contribution_count=len(mine),
            epsilon_spent=epsilon,
            epsilon_remaining=0.0,
            selected_pattern_keys=tuple(sorted(selected)),
            excluded_pattern_keys=(),
            withdrawn=revoked,
        )

    def withdraw(self, *, contributor_id: str, scope_id: str, actor: str | None = None) -> int:
        if not self.authorizer.authorize(scope_id=scope_id, actor=actor, operation="withdraw"):
            raise HiveAggregatorError("withdrawal is not authorized for this scope")
        return self.store.revoke(contributor_id=contributor_id, scope_id=scope_id)

    def _validate_and_decrypt(self, envelope: ContributionEnvelope) -> ContributionPayload:
        try:
            public_key = b64decode(envelope.public_key)
            signature = b64decode(envelope.signature)
        except Exception as exc:
            raise HiveAggregatorError("invalid base64 signing material") from exc
        trusted = self.trusted_keys.get(envelope.key_id)
        if trusted is not None and trusted != public_key:
            raise HiveAggregatorError("publisher key does not match the trusted key id")
        if not verify_signature(
            algorithm="ed25519",
            public_key_bytes=public_key,
            data=envelope.signed_bytes(),
            signature=signature,
        ):
            raise HiveAggregatorError("invalid contribution signature")
        if envelope.payload is not None:
            return envelope.payload
        if (
            self.encryption_key is None
            or envelope.encrypted_payload is None
            or envelope.encryption_nonce is None
        ):
            raise HiveAggregatorError("encrypted contribution has no configured decryption key")
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            plaintext = AESGCM(self.encryption_key).decrypt(
                b64decode(envelope.encryption_nonce), b64decode(envelope.encrypted_payload), None
            )
            return ContributionPayload.from_dict(json.loads(plaintext.decode("utf-8")))
        except Exception as exc:
            raise HiveAggregatorError("encrypted contribution could not be decrypted") from exc

    @staticmethod
    def _scope_id(envelope: ContributionEnvelope) -> str:
        if envelope.payload is not None:
            return str(envelope.payload.scope_id)
        return "encrypted"
