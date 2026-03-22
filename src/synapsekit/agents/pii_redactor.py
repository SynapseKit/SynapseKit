"""PIIRedactor — redact and optionally restore PII in text."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..llm.base import BaseLLM
from .guardrails import PIIDetector


@dataclass
class RedactionResult:
    """Result of a PII redaction operation."""

    redacted_text: str
    mapping: dict[str, str] = field(default_factory=dict)
    pii_types_found: list[str] = field(default_factory=list)


class PIIRedactor:
    """Redact PII from text using the existing ``PIIDetector`` patterns.

    Supports two modes:

    - ``"mask"`` (default) — reversible; replaces PII with placeholders like
      ``[EMAIL_1]`` and stores a mapping so originals can be restored.
    - ``"redact"`` — irreversible; same placeholders but no restore.

    Same-value deduplication: identical PII values always receive the same
    placeholder within a single ``redact()`` call.

    Example::

        redactor = PIIRedactor(pii_types=["email", "phone"])
        result = redactor.redact("Email me at a@b.com or a@b.com, call 555-123-4567")
        # result.redacted_text -> "Email me at [EMAIL_1] or [EMAIL_1], call [PHONE_1]"
        original = redactor.restore(result.redacted_text, result.mapping)
    """

    def __init__(
        self,
        pii_types: list[str] | None = None,
        mode: str = "mask",
    ) -> None:
        if mode not in ("mask", "redact"):
            raise ValueError(f"mode must be 'mask' or 'redact', got {mode!r}")
        self._mode = mode
        self._pii_types = pii_types or list(PIIDetector._PATTERNS.keys())
        self._compiled: dict[str, re.Pattern[str]] = {
            name: re.compile(pattern)
            for name, pattern in PIIDetector._PATTERNS.items()
            if name in self._pii_types
        }

    def redact(self, text: str) -> RedactionResult:
        """Replace PII with numbered placeholders."""
        mapping: dict[str, str] = {}  # placeholder -> original
        value_to_placeholder: dict[str, str] = {}  # original -> placeholder
        counters: dict[str, int] = {}
        pii_types_found: set[str] = set()
        redacted = text

        for pii_type, pattern in self._compiled.items():
            matches = pattern.findall(redacted)
            if not matches:
                continue
            pii_types_found.add(pii_type)
            for match in matches:
                if match in value_to_placeholder:
                    continue
                tag = pii_type.upper()
                counters.setdefault(pii_type, 0)
                counters[pii_type] += 1
                placeholder = f"[{tag}_{counters[pii_type]}]"
                value_to_placeholder[match] = placeholder
                mapping[placeholder] = match

        # Apply replacements (longest values first to avoid partial matches)
        for value in sorted(value_to_placeholder, key=len, reverse=True):
            redacted = redacted.replace(value, value_to_placeholder[value])

        return RedactionResult(
            redacted_text=redacted,
            mapping=mapping if self._mode == "mask" else {},
            pii_types_found=sorted(pii_types_found),
        )

    def restore(self, text: str, mapping: dict[str, str]) -> str:
        """Re-inject original PII values. No-op in ``'redact'`` mode."""
        if self._mode == "redact" or not mapping:
            return text
        result = text
        for placeholder, original in mapping.items():
            result = result.replace(placeholder, original)
        return result

    async def wrap_generate(self, llm: BaseLLM, prompt: str) -> tuple[str, RedactionResult]:
        """Convenience: redact prompt, generate, restore response."""
        redaction = self.redact(prompt)
        response = await llm.generate(redaction.redacted_text)
        restored = self.restore(response, redaction.mapping)
        return restored, redaction
