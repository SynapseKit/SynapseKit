"""Input-stage guards: prompt-injection, jailbreak, topic, and cost gates.

These are heuristic, pattern-based detectors -- fast, dependency-free, and
deterministic. They are a first line of defence, not a guarantee; pair a
``block``-mode injection guard with an output grounding/PII guard for defence in
depth. Patterns are conservative to keep the false-positive rate low.
"""

from __future__ import annotations

import re

from .base import Guard, GuardContext
from .types import GuardFinding, Mode, Stage

# Phrases that try to override the standing instructions or exfiltrate the
# system prompt. Kept specific -- "ignore" alone is far too common in normal text.
_INJECTION_PATTERNS: tuple[str, ...] = (
    r"ignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|messages?|context)",
    r"disregard\s+(all\s+|the\s+)?(previous|prior|above|system)\b",
    r"forget\s+(everything|all|your)\s+(above|previous|instructions?|prompt)",
    r"(reveal|repeat|print|show|output|leak)\s+(me\s+)?(your\s+|the\s+)?(system\s+)?(prompt|instructions?|rules)",
    r"what\s+(are\s+)?your\s+(system\s+)?(prompt|instructions?|initial\s+instructions?)",
    r"override\s+(your\s+)?(instructions?|rules|guardrails?|safety)",
    r"new\s+instructions?\s*:",
)

# Persona / roleplay jailbreaks that try to escape the assistant's guardrails.
_JAILBREAK_PATTERNS: tuple[str, ...] = (
    r"\bDAN\b|do\s+anything\s+now",
    r"developer\s+mode",
    r"jail\s*break",
    r"pretend\s+(you\s+are|to\s+be)\s+.*(no\s+restrictions?|unfiltered|uncensored|evil)",
    r"you\s+(are|have)\s+no\s+(restrictions?|rules|guidelines|limits?)",
    r"act\s+as\s+.*(unfiltered|uncensored|without\s+(any\s+)?restrictions?)",
    r"(bypass|ignore|disable)\s+(your\s+)?(safety|content\s+policy|guardrails?|filters?)",
)


class _RegexInputGuard(Guard):
    """Shared implementation for the heuristic input pattern guards."""

    stage = Stage.INPUT
    _patterns: tuple[str, ...] = ()
    _label = "pattern"

    def __init__(
        self,
        *,
        mode: Mode = Mode.BLOCK,
        name: str | None = None,
        extra_patterns: list[str] | None = None,
    ) -> None:
        super().__init__(mode=mode, name=name)
        self._compiled = [
            re.compile(p, re.IGNORECASE) for p in (*self._patterns, *(extra_patterns or []))
        ]

    async def inspect(self, text: str, context: GuardContext) -> GuardFinding:
        matches = [p.pattern for p in self._compiled if p.search(text)]
        if not matches:
            return self._ok()
        return self._hit(
            f"{self._label} detected ({len(matches)} signal(s))",
            count=len(matches),
            metadata={"signals": matches},
        )


class PromptInjectionGuard(_RegexInputGuard):
    """Flag prompts that try to override standing instructions or exfiltrate the
    system prompt (e.g. "ignore all previous instructions", "print your rules")."""

    _patterns = _INJECTION_PATTERNS
    _label = "prompt injection"


class JailbreakGuard(_RegexInputGuard):
    """Flag persona/roleplay jailbreaks (DAN, "developer mode", "act as an
    unfiltered model", "disable your safety filters")."""

    _patterns = _JAILBREAK_PATTERNS
    _label = "jailbreak attempt"


class TopicGuard(Guard):
    """Allow/deny topics by keyword.

    ``blocked_topics`` triggers if any listed term appears. ``allowed_topics``,
    when non-empty, triggers if *none* of the listed terms appears (an
    allow-list -- stay on approved subjects). Matching is case-insensitive,
    whole-substring.
    """

    stage = Stage.INPUT

    def __init__(
        self,
        *,
        allowed_topics: list[str] | None = None,
        blocked_topics: list[str] | None = None,
        mode: Mode = Mode.BLOCK,
        name: str | None = None,
    ) -> None:
        super().__init__(mode=mode, name=name)
        self._allowed = [t.lower() for t in (allowed_topics or [])]
        self._blocked = [t.lower() for t in (blocked_topics or [])]

    async def inspect(self, text: str, context: GuardContext) -> GuardFinding:
        lowered = text.lower()
        hits = [t for t in self._blocked if t in lowered]
        if hits:
            return self._hit(
                f"blocked topic(s): {', '.join(hits)}",
                count=len(hits),
                metadata={"topics": hits},
            )
        if self._allowed and not any(t in lowered for t in self._allowed):
            return self._hit(
                "off allowed topics",
                metadata={"allowed": self._allowed},
            )
        return self._ok()


class MaxCostGuard(Guard):
    """Reject a call whose estimated cost exceeds ``max_usd``.

    Reads :attr:`GuardContext.estimated_cost_usd`; if the caller did not supply
    an estimate the guard is a no-op (it cannot gate what it cannot see)."""

    stage = Stage.INPUT

    def __init__(
        self,
        max_usd: float,
        *,
        mode: Mode = Mode.BLOCK,
        name: str | None = None,
    ) -> None:
        super().__init__(mode=mode, name=name)
        self.max_usd = max_usd

    async def inspect(self, text: str, context: GuardContext) -> GuardFinding:
        estimate = context.estimated_cost_usd
        if estimate is None or estimate <= self.max_usd:
            return self._ok()
        return self._hit(
            f"estimated cost ${estimate:.4f} exceeds cap ${self.max_usd:.4f}",
            metadata={"estimated_cost_usd": estimate, "max_usd": self.max_usd},
        )
