"""Rule-based trigger policy for ambient interventions.

Deliberately not machine-learned: a small, auditable pattern table covering
the issue's canonical trigger ("about to run something destructive against a
dirty repo"). Add a rule by adding a tuple, not a new abstraction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .events import AmbientEvent, AmbientState

DEFAULT_MIN_CONFIDENCE = 0.6

# (pattern, label, confidence)
_RISKY_PATTERNS: tuple[tuple[re.Pattern[str], str, float], ...] = (
    (
        re.compile(r"\brm\s+-[a-z]*(rf|fr)[a-z]*\b", re.IGNORECASE),
        "destructive-delete",
        0.9,
    ),
    (
        re.compile(r"\bRemove-Item\b.*(-Recurse\b.*-Force\b|-Force\b.*-Recurse\b)", re.IGNORECASE),
        "destructive-delete",
        0.9,
    ),
    (
        re.compile(r"\bgit\s+reset\s+--hard\b", re.IGNORECASE),
        "git-reset-hard",
        0.85,
    ),
    (
        re.compile(r"\bgit\s+(push\s+.*(--force\b|-f\b)|clean\s+-[a-z]*f)", re.IGNORECASE),
        "git-force-push-or-clean",
        0.8,
    ),
)


@dataclass(frozen=True)
class Intervention:
    rule: str
    confidence: float
    message: str
    event: AmbientEvent


def evaluate(event: AmbientEvent, state: AmbientState) -> Intervention | None:
    """Return an Intervention if ``event`` matches a risky pattern against dirty state."""

    if event.source != "terminal" or not state.git_dirty:
        return None

    for pattern, label, confidence in _RISKY_PATTERNS:
        if pattern.search(event.text):
            branch = state.branch or "this repo"
            message = (
                f"'{event.text.strip()}' looks destructive and {branch} has "
                f"{len(state.dirty_files)} uncommitted file(s). "
                "Consider committing or stashing first."
            )
            return Intervention(rule=label, confidence=confidence, message=message, event=event)
    return None
