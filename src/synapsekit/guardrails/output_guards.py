"""Output-stage guards: PII redaction, toxicity, citation/grounding, schema.

These run over the model's response. PII redaction reuses the battle-tested
:class:`~synapsekit.memory.pii_filter.MemoryPIIFilter`; schema validation reuses
Pydantic directly so it needs no extra model call.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from .base import Guard, GuardContext
from .types import GuardFinding, Mode, Stage

if TYPE_CHECKING:
    from pydantic import BaseModel

# A small, conservative default lexicon of slurs/abuse markers. Deliberately
# short and overridable -- toxicity is context-heavy and this is a floor, not a
# classifier. Callers layer their own terms or swap in an LLM-backed guard.
_DEFAULT_TOXIC_TERMS: tuple[str, ...] = (
    "kill yourself",
    "kys",
    "i hate you",
    "you are worthless",
    "you are stupid",
    "go die",
    "retard",
)

# Citation markers: bracketed refs [1], "(source: ...)", "according to", or a URL.
_CITATION_PATTERNS: tuple[str, ...] = (
    r"\[\d+\]",
    r"\(source[:\s]",
    r"\bsources?\s*:",
    r"\baccording to\b",
    r"https?://\S+",
)


class PIIRedactionGuard(Guard):
    """Detect and (by default) redact PII in the model's output.

    Wraps :class:`MemoryPIIFilter`. In ``redact`` mode the finding carries the
    redacted text so the policy substitutes it downstream; in ``block``/``flag``
    mode the presence of PII is reported without rewriting.
    """

    stage = Stage.OUTPUT

    def __init__(
        self,
        *,
        detect: list[str] | None = None,
        mode: Mode = Mode.REDACT,
        name: str | None = None,
    ) -> None:
        super().__init__(mode=mode, name=name)
        from ..memory.pii_filter import MemoryPIIFilter

        self._filter = MemoryPIIFilter(detect=detect, redact=True)

    async def inspect(self, text: str, context: GuardContext) -> GuardFinding:
        result = self._filter.filter_content(text)
        if result.is_clean:
            return self._ok()
        redacted = result.filtered_content if self.mode is Mode.REDACT else None
        return self._hit(
            f"PII detected: {', '.join(result.redaction_types)}",
            count=result.redacted_count,
            redacted_text=redacted,
            metadata={"types": result.redaction_types},
        )


class ToxicityGuard(Guard):
    """Flag toxic/abusive language via a keyword lexicon.

    A deliberately simple, dependency-free floor. Pass ``terms`` to replace the
    default lexicon or ``extra_terms`` to add to it."""

    stage = Stage.OUTPUT

    def __init__(
        self,
        *,
        terms: list[str] | None = None,
        extra_terms: list[str] | None = None,
        mode: Mode = Mode.BLOCK,
        name: str | None = None,
    ) -> None:
        super().__init__(mode=mode, name=name)
        base = terms if terms is not None else list(_DEFAULT_TOXIC_TERMS)
        self._terms = [t.lower() for t in (*base, *(extra_terms or []))]

    async def inspect(self, text: str, context: GuardContext) -> GuardFinding:
        lowered = text.lower()
        hits = [t for t in self._terms if t in lowered]
        if not hits:
            return self._ok()
        return self._hit(
            f"toxic language ({len(hits)} term(s))",
            count=len(hits),
            metadata={"terms": hits},
        )


class CitationRequiredGuard(Guard):
    """Trigger when the output contains no citation/grounding marker.

    Useful for RAG answers that must cite their sources. Recognises bracketed
    refs, "source:"/"according to" phrasing, and inline URLs; supply
    ``patterns`` to override."""

    stage = Stage.OUTPUT

    def __init__(
        self,
        *,
        patterns: list[str] | None = None,
        min_length: int = 40,
        mode: Mode = Mode.FLAG,
        name: str | None = None,
    ) -> None:
        super().__init__(mode=mode, name=name)
        self._min_length = min_length
        self._compiled = [re.compile(p, re.IGNORECASE) for p in (patterns or _CITATION_PATTERNS)]

    async def inspect(self, text: str, context: GuardContext) -> GuardFinding:
        # Trivially short answers (acknowledgements, refusals) are exempt.
        if len(text.strip()) < self._min_length:
            return self._ok()
        if any(p.search(text) for p in self._compiled):
            return self._ok()
        return self._hit("no citation/source found in output")


class SchemaGuard(Guard):
    """Validate that the output parses as JSON and matches a Pydantic model.

    Reuses Pydantic directly -- no extra model call. Triggers when the output is
    not valid JSON or fails schema validation. Defaults to ``block`` mode since a
    contract violation usually means the response is unusable downstream."""

    stage = Stage.OUTPUT

    def __init__(
        self,
        schema: type[BaseModel],
        *,
        mode: Mode = Mode.BLOCK,
        name: str | None = None,
    ) -> None:
        super().__init__(mode=mode, name=name)
        self._schema = schema

    async def inspect(self, text: str, context: GuardContext) -> GuardFinding:
        try:
            data: Any = json.loads(text)
        except (json.JSONDecodeError, ValueError) as exc:
            return self._hit(f"output is not valid JSON: {exc}")
        try:
            self._schema.model_validate(data)
        except Exception as exc:  # pydantic.ValidationError and friends
            return self._hit(f"output failed schema {self._schema.__name__}: {exc}")
        return self._ok()
