"""Core types for the guardrails middleware -- modes, findings, and reports.

A *guard* inspects a piece of text (an inbound prompt or an outbound model
response) and returns a :class:`GuardFinding`. A :class:`GuardrailPolicy`
composes many guards over the ``input`` and ``output`` stages and rolls their
findings up into a single :class:`GuardrailReport` whose :attr:`action` is the
most severe mode that any triggered guard was configured with.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Mode(str, Enum):
    """What a guard does when it triggers.

    ``block`` and ``require_human`` both mark a report as *not allowed*; the
    difference is intent -- ``block`` is a hard refusal, ``require_human`` holds
    the call for out-of-band approval. ``redact`` rewrites the offending text
    and lets the call proceed. ``flag`` records the finding but never alters the
    text or stops the call.
    """

    BLOCK = "block"
    REDACT = "redact"
    FLAG = "flag"
    REQUIRE_HUMAN = "require_human"


class Stage(str, Enum):
    """Whether a guard runs before the model call (``input``) or after it."""

    INPUT = "input"
    OUTPUT = "output"


# Severity ranking used to pick a report's overall action across triggered
# guards. Higher wins; a triggered ``block`` always dominates a ``flag``.
_MODE_SEVERITY: dict[Mode, int] = {
    Mode.FLAG: 1,
    Mode.REDACT: 2,
    Mode.REQUIRE_HUMAN: 3,
    Mode.BLOCK: 4,
}


def strongest_mode(modes: list[Mode]) -> Mode | None:
    """Return the most severe mode in ``modes``, or ``None`` if empty."""
    if not modes:
        return None
    return max(modes, key=lambda m: _MODE_SEVERITY[m])


@dataclass(frozen=True)
class GuardFinding:
    """One guard's verdict on one piece of text."""

    guard: str
    mode: Mode
    triggered: bool
    detail: str = ""
    count: int = 0
    # Set only by redact-capable guards when they actually rewrote the text.
    redacted_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardrailReport:
    """The rolled-up result of running a stage's guards over some text."""

    stage: Stage
    text: str
    allowed: bool
    action: Mode | None = None
    findings: list[GuardFinding] = field(default_factory=list)
    audit_event_id: str | None = None

    @property
    def triggered(self) -> list[GuardFinding]:
        """The findings that actually fired, in guard order."""
        return [f for f in self.findings if f.triggered]

    @property
    def blocked(self) -> bool:
        return self.action is Mode.BLOCK

    @property
    def requires_human(self) -> bool:
        return self.action is Mode.REQUIRE_HUMAN


class GuardrailBlockedError(RuntimeError):
    """Raised by :class:`~synapsekit.guardrails.llm.GuardedLLM` when a guard in
    ``block`` or ``require_human`` mode fires and the call cannot proceed."""

    def __init__(self, report: GuardrailReport) -> None:
        self.report = report
        rules = ", ".join(f.guard for f in report.triggered) or "policy"
        action = report.action.value if report.action else "block"
        super().__init__(f"guardrail {action} at {report.stage.value} stage: {rules}")
