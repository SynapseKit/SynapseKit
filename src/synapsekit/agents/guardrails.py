"""Agent guardrails -- input/output validation for agent safety."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""

    passed: bool
    violations: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        if self.passed:
            return "GuardrailResult(passed=True)"
        return f"GuardrailResult(passed=False, violations={self.violations})"


class ContentFilter:
    """Filter content for blocked patterns or topics.

    Usage::
        filter = ContentFilter(blocked_patterns=[r"password\\s*[:=]", r"\\bSSN\\b"])
        result = filter.check("My password: abc123")
        # result.passed -> False
    """

    def __init__(
        self,
        blocked_patterns: list[str] | None = None,
        blocked_words: list[str] | None = None,
        max_length: int | None = None,
    ) -> None:
        self._patterns = [re.compile(p, re.IGNORECASE) for p in (blocked_patterns or [])]
        self._blocked_words = [w.lower() for w in (blocked_words or [])]
        self._max_length = max_length

    def check(self, text: str) -> GuardrailResult:
        violations: list[str] = []

        for pattern in self._patterns:
            if pattern.search(text):
                violations.append(f"Blocked pattern matched: {pattern.pattern}")

        text_lower = text.lower()
        for word in self._blocked_words:
            if word in text_lower:
                violations.append(f"Blocked word found: {word}")

        if self._max_length and len(text) > self._max_length:
            violations.append(f"Content exceeds max length: {len(text)} > {self._max_length}")

        return GuardrailResult(passed=len(violations) == 0, violations=violations)


class PIIDetector:
    """Detect personally identifiable information.

    Usage::
        detector = PIIDetector()
        result = detector.check("Call me at 555-123-4567 or email john@example.com")
        # result.passed -> False
    """

    _PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    }

    def __init__(self, detect: list[str] | None = None) -> None:
        self._detect = detect or list(self._PATTERNS.keys())
        self._compiled = {
            name: re.compile(pattern)
            for name, pattern in self._PATTERNS.items()
            if name in self._detect
        }

    def check(self, text: str) -> GuardrailResult:
        violations: list[str] = []

        for name, pattern in self._compiled.items():
            matches = pattern.findall(text)
            if matches:
                violations.append(f"PII detected ({name}): {len(matches)} instance(s)")

        return GuardrailResult(passed=len(violations) == 0, violations=violations)


class TopicRestrictor:
    """Restrict agent to specific topics.

    Usage::
        restrictor = TopicRestrictor(
            allowed_topics=["technology", "science"],
            blocked_topics=["politics", "religion"],
        )
    """

    def __init__(
        self,
        allowed_topics: list[str] | None = None,
        blocked_topics: list[str] | None = None,
    ) -> None:
        self._allowed = [t.lower() for t in (allowed_topics or [])]
        self._blocked = [t.lower() for t in (blocked_topics or [])]

    def check(self, text: str) -> GuardrailResult:
        violations: list[str] = []
        text_lower = text.lower()

        for topic in self._blocked:
            if topic in text_lower:
                violations.append(f"Blocked topic detected: {topic}")

        return GuardrailResult(passed=len(violations) == 0, violations=violations)


class Guardrails:
    """Compose multiple guardrail checks.

    Usage::
        guardrails = Guardrails(checks=[
            ContentFilter(blocked_words=["hack", "exploit"]),
            PIIDetector(),
            TopicRestrictor(blocked_topics=["politics"]),
        ])

        result = guardrails.check("My email is john@test.com")
        if not result.passed:
            print(result.violations)
    """

    def __init__(self, checks: list[Any] | None = None) -> None:
        self._checks = checks or []

    def add_check(self, check: Any) -> None:
        self._checks.append(check)

    def check(self, text: str) -> GuardrailResult:
        all_violations: list[str] = []

        for check in self._checks:
            result = check.check(text)
            all_violations.extend(result.violations)

        return GuardrailResult(passed=len(all_violations) == 0, violations=all_violations)
