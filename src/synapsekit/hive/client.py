"""Async Hive client with local-first caching and in-process transport."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, Protocol

from ..audit.signer import Ed25519SigningProvider, SigningProvider
from ..mesh.privacy import MeshPrivacyFilter
from .aggregator import HiveAggregator
from .privacy import (
    DifferentialPrivacy,
    PatternMiner,
    PrivacyBudgetLedger,
    Pseudonymizer,
    stable_scope_id,
)
from .types import (
    ContributionEnvelope,
    ContributionPayload,
    PatternObservation,
    PrivacyConfig,
    ShareScope,
    Suggestion,
    TransparencyReport,
    b64encode,
)


class HiveTransport(Protocol):
    async def submit(self, envelope: ContributionEnvelope) -> str: ...

    async def suggestions(
        self, *, scope_id: str, query: str | None, minimum_cohort: int, limit: int
    ) -> list[Suggestion]: ...

    async def withdraw(self, *, contributor_id: str, scope_id: str) -> int: ...

    async def transparency(self, *, contributor_id: str, scope_id: str) -> TransparencyReport: ...


class InProcessHiveTransport:
    """Adapter used for offline/self-hosted embedded deployments and tests."""

    def __init__(self, aggregator: HiveAggregator) -> None:
        self.aggregator = aggregator

    async def submit(self, envelope: ContributionEnvelope) -> str:
        return await asyncio.to_thread(self.aggregator.submit, envelope)

    async def suggestions(
        self, *, scope_id: str, query: str | None, minimum_cohort: int, limit: int
    ) -> list[Suggestion]:
        return await asyncio.to_thread(
            self.aggregator.suggestions,
            scope_id=scope_id,
            query=query,
            minimum_cohort=minimum_cohort,
            limit=limit,
        )

    async def withdraw(self, *, contributor_id: str, scope_id: str) -> int:
        return await asyncio.to_thread(
            self.aggregator.withdraw, contributor_id=contributor_id, scope_id=scope_id
        )

    async def transparency(self, *, contributor_id: str, scope_id: str) -> TransparencyReport:
        return await asyncio.to_thread(
            self.aggregator.transparency, contributor_id=contributor_id, scope_id=scope_id
        )


class HttpHiveTransport:
    """Dependency-free JSON HTTP transport for the optional FastAPI service."""

    def __init__(self, base_url: str, *, api_key: str | None = None, timeout: float = 20.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    async def submit(self, envelope: ContributionEnvelope) -> str:
        data = await self._request("POST", "/v1/contributions", envelope.to_dict())
        return str(data["contribution_id"])

    async def suggestions(
        self, *, scope_id: str, query: str | None, minimum_cohort: int, limit: int
    ) -> list[Suggestion]:
        path = f"/v1/suggestions?scope_id={_quote(scope_id)}&minimum_cohort={minimum_cohort}&limit={limit}"
        if query:
            path += f"&query={_quote(query)}"
        data = await self._request("GET", path)
        return [Suggestion.from_dict(item) for item in data.get("suggestions", [])]

    async def withdraw(self, *, contributor_id: str, scope_id: str) -> int:
        data = await self._request(
            "POST", "/v1/withdraw", {"contributor_id": contributor_id, "scope_id": scope_id}
        )
        return int(data.get("revoked", 0))

    async def transparency(self, *, contributor_id: str, scope_id: str) -> TransparencyReport:
        data = await self._request(
            "GET",
            f"/v1/transparency?contributor_id={_quote(contributor_id)}&scope_id={_quote(scope_id)}",
        )
        return TransparencyReport(
            scope=ShareScope.parse(str(data["scope"])),
            scope_id=str(data["scope_id"]),
            contribution_count=int(data["contribution_count"]),
            epsilon_spent=float(data["epsilon_spent"]),
            epsilon_remaining=float(data["epsilon_remaining"]),
            selected_pattern_keys=tuple(data.get("selected_pattern_keys", [])),
            excluded_pattern_keys=tuple(data.get("excluded_pattern_keys", [])),
            uploaded_content=bool(data.get("uploaded_content", False)),
            withdrawn=bool(data.get("withdrawn", False)),
        )

    async def _request(
        self, method: str, path: str, body: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        def request() -> dict[str, Any]:
            headers = {"Accept": "application/json"}
            if body is not None:
                headers["Content-Type"] = "application/json"
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            request_object = urllib.request.Request(
                f"{self.base_url}{path}",
                data=json.dumps(body).encode("utf-8") if body is not None else None,
                headers=headers,
                method=method,
            )
            try:
                with urllib.request.urlopen(request_object, timeout=self.timeout) as response:
                    raw = response.read()
            except (urllib.error.HTTPError, urllib.error.URLError) as exc:
                raise HiveClientError(f"Hive service request failed: {exc}") from exc
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, dict):
                raise HiveClientError("Hive service returned a non-object response")
            return value

        return await asyncio.to_thread(request)


class HiveClientError(RuntimeError):
    """Raised for client-side transport, cache, and policy errors."""


class HiveClient:
    """Opt-in client for privacy-preserving shared memory patterns.

    The client owns the sensitive part of the workflow: file selection,
    redaction, bounded vocabulary extraction, pseudonymization, and DP noise
    all happen before ``transport.submit`` is called.
    """

    def __init__(
        self,
        *,
        scope: ShareScope | str = ShareScope.LOCAL,
        team_id: str | None = None,
        contributor_id: str | None = None,
        contribute: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
        privacy: PrivacyConfig | None = None,
        transport: HiveTransport | None = None,
        cache_path: str | Path | None = None,
        signing_provider: SigningProvider | None = None,
        pseudonymizer: Pseudonymizer | None = None,
        encryption_key: bytes | None = None,
        privacy_filter: MeshPrivacyFilter | None = None,
        miner: PatternMiner | None = None,
        dp: DifferentialPrivacy | None = None,
    ) -> None:
        self.scope = ShareScope.parse(scope)
        self.scope_id = stable_scope_id(self.scope.value, team_id)
        self.config = privacy or PrivacyConfig()
        self.include = tuple(contribute or ("**/*.md", "*.md"))
        self.exclude = tuple(exclude or ())
        self.cache_path = Path(cache_path or Path.home() / ".synapsekit" / "hive.json")
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.signing_provider = signing_provider or Ed25519SigningProvider()
        self.pseudonymizer = pseudonymizer or self._load_or_create_pseudonymizer()
        self.encryption_key = encryption_key
        if encryption_key is not None and len(encryption_key) not in {16, 24, 32}:
            raise ValueError("encryption_key must be 16, 24, or 32 bytes for AES-GCM")
        self.contributor_id = self.pseudonymizer.pseudonym(
            contributor_id or secrets.token_hex(16), self.scope_id
        )
        self.privacy_filter = privacy_filter or MeshPrivacyFilter()
        self.miner = miner or PatternMiner()
        self.dp = dp or DifferentialPrivacy()
        self._ledger = self._load_ledger()
        self.transport = transport
        self._last_patterns: tuple[str, ...] = ()
        self._last_excluded: tuple[str, ...] = ()

    async def contribute(self, roots: Iterable[str | Path] = (".",)) -> str:
        self._ledger.reserve()
        mined = await asyncio.to_thread(
            self.miner.mine,
            roots,
            include=self.include,
            exclude=self.exclude,
            privacy_filter=self.privacy_filter,
            max_patterns=self.config.max_patterns,
        )
        self._last_patterns = mined.selected_keys
        self._last_excluded = mined.excluded_keys
        counts = {pattern.key: pattern.value for pattern in mined.patterns}
        privatized = self.dp.privatize(__import__("collections").Counter(counts), self.config)
        patterns = tuple(
            PatternObservation(
                key=pattern.key, value=privatized[pattern.key], category=pattern.category
            )
            for pattern in mined.patterns
        )
        payload = ContributionPayload(
            scope=self.scope,
            scope_id=self.scope_id,
            patterns=patterns,
            epsilon=self.config.epsilon,
            delta=self.config.delta,
        )
        envelope = self._sign(payload)
        self._save_ledger()
        if self.transport is None:
            raise HiveClientError(
                "no Hive transport configured; contribution remains local and unuploaded"
            )
        try:
            return await self.transport.submit(envelope)
        except Exception:
            # Keep the budget reservation: retrying the same source would spend
            # another privacy budget in a real deployment.
            raise

    async def suggestions_for(
        self, query: str | None = None, *, limit: int = 20
    ) -> list[Suggestion]:
        cached = self._read_cache().get("suggestions", [])
        if self.transport is None:
            return [Suggestion.from_dict(item) for item in cached][:limit]
        try:
            suggestions = await self.transport.suggestions(
                scope_id=self.scope_id,
                query=query,
                minimum_cohort=self.config.minimum_cohort,
                limit=limit,
            )
        except Exception:
            return [Suggestion.from_dict(item) for item in cached][:limit]
        self._write_cache({"suggestions": [item.to_dict() for item in suggestions]})
        return suggestions

    async def withdraw(self) -> int:
        if self.transport is None:
            raise HiveClientError("no Hive transport configured")
        count = await self.transport.withdraw(
            contributor_id=self.contributor_id, scope_id=self.scope_id
        )
        self._write_cache({"suggestions": [], "withdrawn": True})
        return count

    async def transparency(self) -> TransparencyReport:
        if self.transport is not None:
            try:
                return await self.transport.transparency(
                    contributor_id=self.contributor_id, scope_id=self.scope_id
                )
            except Exception:
                pass
        return TransparencyReport(
            scope=self.scope,
            scope_id=self.scope_id,
            contribution_count=self._ledger.contribution_count,
            epsilon_spent=self._ledger.spent,
            epsilon_remaining=self._ledger.remaining,
            selected_pattern_keys=self._last_patterns,
            excluded_pattern_keys=self._last_excluded,
        )

    def _sign(self, payload: ContributionPayload) -> ContributionEnvelope:
        if self.encryption_key is not None:
            try:
                from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            except ImportError as exc:  # pragma: no cover - cryptography is a hard dependency
                raise HiveClientError(
                    "cryptography is required for encrypted Hive payloads"
                ) from exc
            nonce = os.urandom(12)
            ciphertext = AESGCM(self.encryption_key).encrypt(nonce, payload.canonical_bytes(), None)
            signed_value = {"ciphertext": b64encode(ciphertext), "nonce": b64encode(nonce)}
            signature = self.signing_provider.sign(
                json.dumps(signed_value, sort_keys=True, separators=(",", ":")).encode("utf-8")
            )
            return ContributionEnvelope(
                payload=None,
                contributor_id=self.contributor_id,
                public_key=b64encode(self.signing_provider.public_key_bytes()),
                signature=b64encode(signature),
                encrypted_payload=signed_value["ciphertext"],
                encryption_nonce=signed_value["nonce"],
            )
        signature = self.signing_provider.sign(payload.canonical_bytes())
        return ContributionEnvelope(
            payload=payload,
            contributor_id=self.contributor_id,
            public_key=b64encode(self.signing_provider.public_key_bytes()),
            signature=b64encode(signature),
        )

    def _load_or_create_pseudonymizer(self) -> Pseudonymizer:
        try:
            value = json.loads(self.cache_path.read_text(encoding="utf-8"))
            secret = value.get("pseudonymizer_secret") if isinstance(value, dict) else None
            if isinstance(secret, str):
                return Pseudonymizer(bytes.fromhex(secret))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        pseudonymizer = Pseudonymizer()
        self._write_cache({"pseudonymizer_secret": pseudonymizer.secret.hex()})
        return pseudonymizer

    def _load_ledger(self) -> PrivacyBudgetLedger:
        try:
            value = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and isinstance(value.get("ledger"), dict):
                return PrivacyBudgetLedger.from_dict(self.config, value["ledger"])
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        return PrivacyBudgetLedger(self.config)

    def _save_ledger(self) -> None:
        self._write_cache({"ledger": self._ledger.to_dict()})

    def _read_cache(self) -> dict[str, Any]:
        try:
            value = json.loads(self.cache_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return {}

    def _write_cache(self, values: Mapping[str, Any]) -> None:
        current = self._read_cache()
        current.update(values)
        temporary = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        temporary.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.cache_path)
        with suppress(OSError):
            os.chmod(self.cache_path, 0o600)


def _quote(value: str) -> str:
    from urllib.parse import quote

    return quote(value, safe="")
