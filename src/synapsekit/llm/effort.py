from __future__ import annotations

import re
from collections.abc import Callable
from enum import Enum
from typing import Literal


class EffortLevel(str, Enum):
    """Provider-independent reasoning effort levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


EffortInput = EffortLevel | Literal["auto"] | str
EffortClassifier = Callable[[str], EffortLevel | str]

GEMINI_THINKING_BUDGETS: dict[EffortLevel, int] = {
    EffortLevel.LOW: 512,
    EffortLevel.MEDIUM: 1024,
    EffortLevel.HIGH: 2048,
}

_WORD_RE = re.compile(r"\b[\w]+(?:[-'][\w]+)*\b")
_HIGH_EFFORT_RE = re.compile(
    r"\b(?:analy[sz]e|architect|compare|debug|derive|design|evaluate|optimi[sz]e|"
    r"prove|proof|trade[- ]offs?|multi[- ]step|step by step|root cause|complexity)\b"
)
_MEDIUM_EFFORT_RE = re.compile(r"\b(?:explain|summari[sz]e|example|why|how)\b")


def normalize_effort(effort: EffortInput | None) -> EffortLevel | Literal["auto"] | None:
    """Validate and normalize an explicit effort selection."""
    if effort is None:
        return None
    if effort == "auto":
        return "auto"
    try:
        return EffortLevel(effort)
    except (TypeError, ValueError):
        raise ValueError("effort must be EffortLevel low/medium/high or 'auto'") from None


def classify_effort(prompt: str) -> EffortLevel:
    """Return a cheap, deterministic difficulty estimate without network calls."""
    normalized_prompt = prompt.casefold()
    word_count = len(_WORD_RE.findall(normalized_prompt))
    if _HIGH_EFFORT_RE.search(normalized_prompt) or word_count >= 70:
        return EffortLevel.HIGH
    if _MEDIUM_EFFORT_RE.search(normalized_prompt) or word_count > 20:
        return EffortLevel.MEDIUM
    return EffortLevel.LOW


def resolve_effort(
    effort: EffortInput | None,
    prompt: str,
    classifier: EffortClassifier | None = None,
) -> EffortLevel | None:
    """Resolve an explicit or auto-classified effort for one prompt."""
    normalized = normalize_effort(effort)
    if normalized is None:
        return None
    if normalized == "auto":
        classify = classifier or classify_effort
        classified = classify(prompt)
        try:
            return EffortLevel(classified)
        except (TypeError, ValueError):
            raise ValueError(
                "effort_classifier must return EffortLevel low, medium, or high"
            ) from None
    return normalized
