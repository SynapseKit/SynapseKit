"""Versioned data types for SynapseKit Hive Mode.

Hive deliberately has a small, JSON-compatible wire format.  The payload is
made from aggregate-friendly pattern observations rather than memory text.
That distinction is important: differential privacy reduces the risk of an
individual observation, but it is not a licence to upload the observation's
original content.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

HIVE_SCHEMA_VERSION = "1.0"


class ShareScope(str, Enum):
    """The audience that can receive a Hive aggregate."""

    LOCAL = "local"
    TEAM = "team"
    COMMUNITY = "community"

    @classmethod
    def parse(cls, value: str | ShareScope) -> ShareScope:
        if isinstance(value, cls):
            return value
        try:
            return cls(value.lower().strip())
        except ValueError as exc:
            raise ValueError("scope must be local, team, or community") from exc


@dataclass(frozen=True, slots=True)
class PrivacyConfig:
    """Local differential-privacy policy for one contribution window."""

    epsilon: float = 1.0
    delta: float = 1e-6
    budget_limit: float = 10.0
    minimum_cohort: int = 3
    max_patterns: int = 64
    max_contributions_per_day: int = 10
    noise_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.epsilon <= 0 or self.budget_limit <= 0:
            raise ValueError("epsilon and budget_limit must be positive")
        if not 0 < self.delta < 1:
            raise ValueError("delta must be between 0 and 1")
        if self.epsilon > self.budget_limit:
            raise ValueError("epsilon cannot exceed the total budget_limit")
        if self.minimum_cohort < 1:
            raise ValueError("minimum_cohort must be at least 1")
        if self.max_patterns < 1 or self.max_contributions_per_day < 1:
            raise ValueError("pattern and contribution limits must be positive")
        if self.noise_scale <= 0:
            raise ValueError("noise_scale must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "budget_limit": self.budget_limit,
            "minimum_cohort": self.minimum_cohort,
            "max_patterns": self.max_patterns,
            "max_contributions_per_day": self.max_contributions_per_day,
            "noise_scale": self.noise_scale,
        }


@dataclass(frozen=True, slots=True)
class PatternObservation:
    """A safe, DP-noised convention observation.

    ``key`` is intentionally a vocabulary item such as
    ``practice:exponential-backoff``.  It is never a filename, path, heading,
    URL, identifier, or excerpt from a user's memory.
    """

    key: str
    value: float
    category: str = "practice"

    def __post_init__(self) -> None:
        if not self.key or len(self.key) > 160:
            raise ValueError("pattern keys must be non-empty and bounded")
        if self.category not in {"framework", "practice", "tooling", "memory"}:
            raise ValueError("unsupported pattern category")
        if self.value != self.value or abs(self.value) > 1_000_000:
            raise ValueError("pattern value must be finite and bounded")

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "value": round(float(self.value), 8), "category": self.category}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PatternObservation:
        return cls(
            key=str(value["key"]),
            value=float(value["value"]),
            category=str(value.get("category", "practice")),
        )


@dataclass(frozen=True, slots=True)
class ContributionPayload:
    """The only semantic data that a contributor sends to Hive."""

    scope: ShareScope
    scope_id: str
    patterns: tuple[PatternObservation, ...]
    epsilon: float
    delta: float
    generated_at: float = field(default_factory=time.time)
    contributor_nonce: str = field(default_factory=lambda: uuid.uuid4().hex)
    extractor_version: str = "1"

    def __post_init__(self) -> None:
        if not self.scope_id:
            raise ValueError("scope_id is required")
        if self.scope is ShareScope.LOCAL and self.scope_id != "local":
            raise ValueError("local contributions must use the local scope id")
        if len(self.patterns) > 256:
            raise ValueError("contribution contains too many patterns")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HIVE_SCHEMA_VERSION,
            "scope": self.scope.value,
            "scope_id": self.scope_id,
            "patterns": [pattern.to_dict() for pattern in self.patterns],
            "epsilon": self.epsilon,
            "delta": self.delta,
            "generated_at": self.generated_at,
            "contributor_nonce": self.contributor_nonce,
            "extractor_version": self.extractor_version,
        }

    def canonical_bytes(self) -> bytes:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ContributionPayload:
        if value.get("schema_version") != HIVE_SCHEMA_VERSION:
            raise ValueError("unsupported Hive contribution schema")
        return cls(
            scope=ShareScope.parse(str(value["scope"])),
            scope_id=str(value["scope_id"]),
            patterns=tuple(
                PatternObservation.from_dict(item) for item in value.get("patterns", [])
            ),
            epsilon=float(value["epsilon"]),
            delta=float(value["delta"]),
            generated_at=float(value["generated_at"]),
            contributor_nonce=str(value["contributor_nonce"]),
            extractor_version=str(value.get("extractor_version", "1")),
        )


@dataclass(frozen=True, slots=True)
class ContributionEnvelope:
    """Signed contribution envelope stored by an aggregator."""

    payload: ContributionPayload | None
    contributor_id: str
    public_key: str
    signature: str
    key_id: str = ""
    encrypted_payload: str | None = None
    encryption_nonce: str | None = None
    received_at: float = field(default_factory=time.time)
    revoked: bool = False

    @property
    def contribution_id(self) -> str:
        raw = f"{self.contributor_id}:{self.signature}".encode()
        return hashlib.sha256(raw).hexdigest()

    def signed_bytes(self) -> bytes:
        if self.encrypted_payload is not None:
            value = {"ciphertext": self.encrypted_payload, "nonce": self.encryption_nonce}
        elif self.payload is not None:
            value = self.payload.to_dict()
        else:
            raise ValueError("envelope has neither payload nor ciphertext")
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HIVE_SCHEMA_VERSION,
            "contribution_id": self.contribution_id,
            "payload": self.payload.to_dict() if self.payload is not None else None,
            "contributor_id": self.contributor_id,
            "public_key": self.public_key,
            "signature": self.signature,
            "key_id": self.key_id,
            "encrypted_payload": self.encrypted_payload,
            "encryption_nonce": self.encryption_nonce,
            "received_at": self.received_at,
            "revoked": self.revoked,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ContributionEnvelope:
        raw_payload = value.get("payload")
        return cls(
            payload=ContributionPayload.from_dict(raw_payload)
            if isinstance(raw_payload, Mapping)
            else None,
            contributor_id=str(value["contributor_id"]),
            public_key=str(value["public_key"]),
            signature=str(value["signature"]),
            key_id=str(value.get("key_id", "")),
            encrypted_payload=value.get("encrypted_payload"),
            encryption_nonce=value.get("encryption_nonce"),
            received_at=float(value.get("received_at", time.time())),
            revoked=bool(value.get("revoked", False)),
        )


@dataclass(frozen=True, slots=True)
class Suggestion:
    """An aggregate-only recommendation returned to a client."""

    key: str
    category: str
    statement: str
    prevalence: float
    confidence: float
    contributor_count: int
    cohort_size: int
    scope: ShareScope

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "category": self.category,
            "statement": self.statement,
            "prevalence": round(self.prevalence, 6),
            "confidence": round(self.confidence, 6),
            "contributor_count": self.contributor_count,
            "cohort_size": self.cohort_size,
            "scope": self.scope.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Suggestion:
        return cls(
            key=str(value["key"]),
            category=str(value["category"]),
            statement=str(value["statement"]),
            prevalence=float(value["prevalence"]),
            confidence=float(value["confidence"]),
            contributor_count=int(value["contributor_count"]),
            cohort_size=int(value["cohort_size"]),
            scope=ShareScope.parse(str(value["scope"])),
        )


@dataclass(frozen=True, slots=True)
class TransparencyReport:
    """User-facing explanation of what was contributed after privacy steps."""

    scope: ShareScope
    scope_id: str
    contribution_count: int
    epsilon_spent: float
    epsilon_remaining: float
    selected_pattern_keys: tuple[str, ...]
    excluded_pattern_keys: tuple[str, ...]
    uploaded_content: bool = False
    withdrawn: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope.value,
            "scope_id": self.scope_id,
            "contribution_count": self.contribution_count,
            "epsilon_spent": round(self.epsilon_spent, 8),
            "epsilon_remaining": round(self.epsilon_remaining, 8),
            "selected_pattern_keys": list(self.selected_pattern_keys),
            "excluded_pattern_keys": list(self.excluded_pattern_keys),
            "uploaded_content": self.uploaded_content,
            "withdrawn": self.withdrawn,
        }


def b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii")


def b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value.encode("ascii"))
