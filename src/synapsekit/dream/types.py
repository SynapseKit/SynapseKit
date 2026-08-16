"""Public data contracts for local-first Dream Mode runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

DreamTask = Literal[
    "distill_lessons",
    "propose_memory_patches",
    "consolidate_entities",
    "prune_stale",
]
DreamRunStatus = Literal["completed", "skipped", "failed"]

DEFAULT_TASKS: tuple[DreamTask, ...] = (
    "distill_lessons",
    "propose_memory_patches",
    "consolidate_entities",
    "prune_stale",
)


@dataclass(frozen=True)
class PowerStatus:
    """Power information used by the overnight safety gate.

    ``known=False`` is intentional: a Dream Mode run fails closed when the
    platform cannot report whether the machine is plugged in.
    """

    plugged_in: bool
    battery_percent: int | None = None
    known: bool = True


@dataclass(frozen=True)
class DreamConfig:
    """Resource and policy settings for :class:`~synapsekit.dream.DreamMode`."""

    schedule: str = "idle_30m or 02:00"
    budget_tokens: int = 100_000
    tasks: tuple[DreamTask, ...] = DEFAULT_TASKS
    lookback_hours: float = 24.0
    stale_after_days: int = 90
    max_trace_chars: int = 120_000
    idle_after_seconds: float = 30 * 60
    poll_seconds: float = 60.0
    require_plugged_in: bool = True
    state_path: str | Path = "~/.synapsekit/dream/state.sqlite3"
    audit_dir: str | Path = "~/.synapsekit/dream/audit"
    # Path to a persisted Ed25519 signing key. When ``None`` (the default),
    # DreamMode uses ``<state_path parent>/signing_key`` so every night's
    # bundle is signed by the same, pinnable per-install key (attestable).
    # Pass an explicit ``signing_policy`` to DreamMode to override entirely
    # (BYOK/KMS, or an ephemeral key).
    signing_key_path: str | Path | None = None

    def __post_init__(self) -> None:
        if self.budget_tokens <= 0:
            raise ValueError("budget_tokens must be positive")
        if self.lookback_hours <= 0:
            raise ValueError("lookback_hours must be positive")
        if self.stale_after_days <= 0:
            raise ValueError("stale_after_days must be positive")
        if self.max_trace_chars <= 0:
            raise ValueError("max_trace_chars must be positive")
        if self.idle_after_seconds < 0:
            raise ValueError("idle_after_seconds cannot be negative")
        if self.poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        unknown = set(self.tasks) - set(DEFAULT_TASKS)
        if unknown:
            raise ValueError(f"unknown Dream Mode task(s): {sorted(unknown)}")


@dataclass(frozen=True)
class Lesson:
    """A distilled, evidence-linked lesson from one or more traces."""

    text: str
    theme: str
    confidence: float
    evidence_event_ids: tuple[str, ...] = ()
    corrections: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("lesson text must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("lesson confidence must be between 0 and 1")


@dataclass(frozen=True)
class StaleMemory:
    """A memory file flagged for review; Dream Mode never deletes it."""

    path: str
    last_read_at: str | None
    last_modified_at: str | None
    age_days: int
    reason: str


@dataclass(frozen=True)
class MeshConsolidation:
    """A duplicate/entity candidate surfaced for human review."""

    left_path: str
    right_path: str
    score: float
    reason: str


@dataclass
class DreamRunResult:
    """Serializable morning-briefing result for one Dream Mode run."""

    run_id: str
    status: DreamRunStatus
    started_at: str
    completed_at: str
    traces_replayed: int = 0
    lessons: list[Lesson] = field(default_factory=list)
    patch_ids: list[str] = field(default_factory=list)
    mesh_reindexed: bool = False
    mesh_consolidations: list[MeshConsolidation] = field(default_factory=list)
    stale_memories: list[StaleMemory] = field(default_factory=list)
    estimated_tokens: int = 0
    audit_path: str | None = None
    audit_key_id: str | None = None
    audit_attestable: bool = False
    warnings: list[str] = field(default_factory=list)
    skipped_reason: str | None = None

    @classmethod
    def skipped(cls, reason: str, *, now: datetime | None = None) -> DreamRunResult:
        timestamp = (now or datetime.now(UTC)).isoformat()
        return cls(
            run_id="",
            status="skipped",
            started_at=timestamp,
            completed_at=timestamp,
            skipped_reason=reason,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly report suitable for a morning briefing."""

        return asdict(self)


@dataclass(frozen=True)
class TraceWindow:
    """The bounded local time window replayed by Dream Mode."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("trace window datetimes must be timezone-aware")
        if self.end < self.start:
            raise ValueError("trace window end must not precede start")
