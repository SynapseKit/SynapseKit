"""PII filtering for Living Memory patches."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..agents.guardrails import GuardrailResult, PIIDetector


@dataclass
class PIIFilterResult:
    """Result of PII filtering on a memory patch.

    When ``is_clean`` is True the content contained no detectable PII.
    Otherwise ``filtered_content`` holds the redacted version and
    ``redaction_types`` lists which categories were found.
    """

    is_clean: bool
    original_content: str
    filtered_content: str
    redacted_count: int = 0
    redaction_types: list[str] = field(default_factory=list)


class MemoryPIIFilter:
    """Filter PII from proposed memory patches before they reach storage.

    Leverages the existing :class:`PIIDetector` from the guardrails module
    and adds active redaction — replacing detected PII with placeholder
    tokens so that memory files never contain sensitive information.

    Parameters
    ----------
    detect:
        List of PII types to scan for.  Defaults to all known types.
    redact:
        When True (default), detected PII is replaced with placeholders.
        When False, the filter only reports findings without modifying content.
    """

    _REDACTION_PATTERNS: dict[str, tuple[str, str]] = {
        "email": (
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "[REDACTED_EMAIL]",
        ),
        "phone": (
            r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "[REDACTED_PHONE]",
        ),
        "ssn": (
            r"\b\d{3}-\d{2}-\d{4}\b",
            "[REDACTED_SSN]",
        ),
        "credit_card": (
            r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
            "[REDACTED_CC]",
        ),
        "ip_address": (
            r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
            "[REDACTED_IP]",
        ),
        "api_key": (
            r"\b(?:sk|pk|api|key|token|secret)[-_]?[A-Za-z0-9]{20,}\b",
            "[REDACTED_KEY]",
        ),
    }

    def __init__(
        self,
        detect: list[str] | None = None,
        *,
        redact: bool = True,
    ) -> None:
        active_types = detect or list(self._REDACTION_PATTERNS.keys())
        self._detector = PIIDetector(
            detect=[t for t in active_types if t in PIIDetector._PATTERNS]
        )
        self._redact = redact
        self._compiled: dict[str, tuple[re.Pattern[str], str]] = {}
        for name, (pattern, replacement) in self._REDACTION_PATTERNS.items():
            if name in active_types:
                self._compiled[name] = (re.compile(pattern), replacement)

    def check(self, text: str) -> GuardrailResult:
        """Check for PII without redacting."""
        return self._detector.check(text)

    def filter_content(self, content: str) -> PIIFilterResult:
        """Detect and optionally redact PII from content."""
        check_result = self._detector.check(content)

        if check_result.passed:
            return PIIFilterResult(
                is_clean=True,
                original_content=content,
                filtered_content=content,
            )

        if not self._redact:
            return PIIFilterResult(
                is_clean=False,
                original_content=content,
                filtered_content=content,
                redaction_types=[
                    v.split("(")[1].rstrip(")").split(":")[0]
                    for v in check_result.violations
                ],
            )

        # Apply redactions
        filtered = content
        total_redacted = 0
        types_found: list[str] = []

        for pii_type, (pattern, replacement) in self._compiled.items():
            matches = pattern.findall(filtered)
            if matches:
                filtered = pattern.sub(replacement, filtered)
                total_redacted += len(matches)
                types_found.append(pii_type)

        return PIIFilterResult(
            is_clean=False,
            original_content=content,
            filtered_content=filtered,
            redacted_count=total_redacted,
            redaction_types=types_found,
        )
