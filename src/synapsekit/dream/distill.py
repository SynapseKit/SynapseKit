"""Bounded, local-first lesson distillation for Dream Mode."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Protocol

from ..audit import AuditRecord, EventKind
from .types import Lesson

_SIGNAL_RE = re.compile(r"(?i)\b(correct|correction|wrong|failed|failure|fix|should|retry|error)\b")


class LessonBackend(Protocol):
    """Minimal async generation contract accepted by Dream Mode."""

    async def generate(self, prompt: str, **kwargs: Any) -> str: ...


def estimate_tokens(text: str) -> int:
    """Use a conservative, dependency-free token estimate."""

    compact = " ".join(text.split())
    return max(0, (len(compact) + 3) // 4)


def trace_transcript(records: Iterable[AuditRecord], *, max_chars: int) -> str:
    """Serialize trace evidence without including full arbitrary objects."""

    lines: list[str] = []
    used = 0
    for record in records:
        payload = json.dumps(record.payload, sort_keys=True, default=str)
        line = f"[{record.timestamp.isoformat()}] {record.kind} {record.event_id}: {payload}"
        remaining = max_chars - used
        if remaining <= 0:
            break
        lines.append(line[:remaining])
        used += min(len(line), remaining) + 1
    return "\n".join(lines)


class DeterministicLessonDistiller:
    """Extract repeatable lessons when no model is configured.

    It intentionally surfaces only error/correction-shaped evidence.  This
    keeps the offline fallback useful while avoiding speculative memory writes.
    """

    def distill(self, records: list[AuditRecord], *, max_lessons: int = 20) -> list[Lesson]:
        groups: dict[str, list[tuple[AuditRecord, str]]] = defaultdict(list)
        for record in records:
            text = self._signal_text(record)
            if not text or not _SIGNAL_RE.search(text):
                continue
            key = self._normalise(text)
            groups[key].append((record, text))

        lessons: list[Lesson] = []
        for key, evidence in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
            records_for_lesson = [record for record, _ in evidence]
            examples = tuple(text[:300] for _, text in evidence[:3])
            confidence = min(0.98, 0.55 + 0.12 * min(len(evidence), 3))
            lessons.append(
                Lesson(
                    text=f"Review and retain this recurring correction: {examples[0]}",
                    theme=key,
                    confidence=confidence,
                    evidence_event_ids=tuple(record.event_id for record in records_for_lesson[:8]),
                    corrections=examples,
                )
            )
            if len(lessons) >= max_lessons:
                break
        return lessons

    @staticmethod
    def _signal_text(record: AuditRecord) -> str:
        payload = record.payload
        candidates = [
            payload.get("error"),
            payload.get("message"),
            payload.get("output"),
            payload.get("reason"),
            payload.get("content"),
        ]
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
            if isinstance(candidate, dict):
                nested = candidate.get("error") or candidate.get("message")
                if isinstance(nested, str) and nested.strip():
                    return nested.strip()
        if record.kind == EventKind.ERROR.value:
            return json.dumps(payload, sort_keys=True, default=str)
        return ""

    @staticmethod
    def _normalise(text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()[:120]


class ModelLessonDistiller:
    """Ask an injected local-first backend for structured lessons.

    A malformed or empty response never blocks a run: deterministic evidence
    extraction remains the safety fallback.
    """

    def __init__(
        self, backend: LessonBackend, fallback: DeterministicLessonDistiller | None = None
    ):
        self.backend = backend
        self.fallback = fallback or DeterministicLessonDistiller()

    async def distill(
        self,
        records: list[AuditRecord],
        *,
        max_chars: int,
        max_lessons: int = 20,
    ) -> tuple[list[Lesson], int]:
        transcript = trace_transcript(records, max_chars=max_chars)
        if not transcript:
            return [], 0
        prompt = (
            "You are Dream Mode's offline memory consolidator. Analyze the trace below and "
            "return only a JSON array. Each item must contain text, theme, confidence "
            "(0..1), evidence_event_ids, and corrections. Surface durable corrections and "
            "repeated failures; omit guesses and one-off noise.\n\nTRACE:\n" + transcript
        )
        try:
            raw = await self.backend.generate(prompt, max_tokens=max_lessons * 120)
            lessons = self._parse(raw, max_lessons=max_lessons)
        except Exception:
            lessons = []
        if not lessons:
            lessons = self.fallback.distill(records, max_lessons=max_lessons)
        return lessons, estimate_tokens(prompt)

    @staticmethod
    def _parse(raw: str, *, max_lessons: int) -> list[Lesson]:
        cleaned = str(raw).strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").removeprefix("json").strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            start, end = cleaned.find("["), cleaned.rfind("]")
            if start < 0 or end <= start:
                return []
            try:
                parsed = json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                return []
        if not isinstance(parsed, list):
            return []
        lessons: list[Lesson] = []
        for item in parsed:
            if not isinstance(item, dict) or not str(item.get("text", "")).strip():
                continue
            try:
                lessons.append(
                    Lesson(
                        text=str(item["text"]).strip()[:1000],
                        theme=str(item.get("theme", "general")).strip()[:120] or "general",
                        confidence=max(0.0, min(1.0, float(item.get("confidence", 0.5)))),
                        evidence_event_ids=tuple(
                            str(value) for value in item.get("evidence_event_ids", [])
                        )[:8],
                        corrections=tuple(
                            str(value)[:300] for value in item.get("corrections", [])
                        )[:5],
                    )
                )
            except (TypeError, ValueError):
                continue
            if len(lessons) >= max_lessons:
                break
        return lessons
