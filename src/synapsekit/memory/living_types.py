"""Data types for Living Memory — file-level patches with signatures."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

PatchStatus = Literal[
    "pending",
    "approved",
    "applied",
    "rejected",
    "reverted",
    "conflict",
]

MemoryFileCategory = Literal[
    "user",
    "feedback",
    "project",
    "general",
]


@dataclass
class MemoryPatch:
    """A proposed diff to a memory file, with provenance and signature.

    Each patch captures the full before/after content of the target file,
    along with a unified diff, rationale for the change, evidence from
    the originating session, and a SHA-256 signature for integrity.
    """

    file_path: str
    before_content: str
    after_content: str
    unified_diff: str
    rationale: str
    evidence_refs: list[str] = field(default_factory=list)
    patch_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    status: PatchStatus = "pending"
    category: MemoryFileCategory = "general"
    session_id: str | None = None
    author: str = "living-memory"
    signature: str = ""
    applied_at: str | None = None
    reverted_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def sign(self, secret: str = "") -> str:
        """Compute and store a SHA-256 signature over patch content."""
        payload = self._signature_material()
        raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
        self.signature = hashlib.sha256(f"{secret}:{raw}".encode()).hexdigest()
        return self.signature

    def verify(self, secret: str = "") -> bool:
        """Check that the stored signature matches the current content."""
        payload = self._signature_material()
        raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
        expected = hashlib.sha256(f"{secret}:{raw}".encode()).hexdigest()
        return self.signature == expected

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryPatch:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def _signature_material(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("signature", None)
        data.pop("applied_at", None)
        data.pop("reverted_at", None)
        return data


@dataclass
class OccurrenceRecord:
    """Tracks how many times a particular fact or pattern has been observed.

    Used by the occurrence threshold system to prevent memory pollution
    from single observations — a fact must be seen at least N times
    across different sessions before a patch is proposed.
    """

    fact_key: str
    count: int = 0
    first_seen: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_seen: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    session_ids: list[str] = field(default_factory=list)
    sample_evidence: list[str] = field(default_factory=list)

    def increment(self, session_id: str, evidence: str = "") -> None:
        """Record a new observation of this fact."""
        self.count += 1
        self.last_seen = datetime.now(timezone.utc).isoformat()
        if session_id and session_id not in self.session_ids:
            self.session_ids.append(session_id)
        if evidence and len(self.sample_evidence) < 5:
            self.sample_evidence.append(evidence[:500])
