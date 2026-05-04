from __future__ import annotations

import time
from dataclasses import dataclass

# Cost per token in USD (input, output)
COST_TABLE: dict[str, dict[str, float]] = {
    # OpenAI — GPT-4o family
    "gpt-4o-mini": {"input": 0.15 / 1e6, "output": 0.60 / 1e6},
    "gpt-4o": {"input": 2.50 / 1e6, "output": 10.00 / 1e6},
    "gpt-4o-2024-11-20": {"input": 2.50 / 1e6, "output": 10.00 / 1e6},
    "gpt-4-turbo": {"input": 10.00 / 1e6, "output": 30.00 / 1e6},
    # OpenAI — GPT-4.1 family
    "gpt-4.1": {"input": 2.00 / 1e6, "output": 8.00 / 1e6},
    "gpt-4.1-mini": {"input": 0.40 / 1e6, "output": 1.60 / 1e6},
    "gpt-4.1-nano": {"input": 0.10 / 1e6, "output": 0.40 / 1e6},
    # OpenAI — o-series reasoning
    "o3": {"input": 10.00 / 1e6, "output": 40.00 / 1e6},
    "o3-mini": {"input": 1.10 / 1e6, "output": 4.40 / 1e6},
    "o4-mini": {"input": 1.10 / 1e6, "output": 4.40 / 1e6},
    # Anthropic
    "claude-haiku-4-5-20251001": {"input": 0.80 / 1e6, "output": 4.00 / 1e6},
    "claude-sonnet-4-6": {"input": 3.00 / 1e6, "output": 15.00 / 1e6},
    "claude-opus-4-6": {"input": 15.00 / 1e6, "output": 75.00 / 1e6},
    # Google Gemini
    "gemini-2.5-pro": {"input": 1.25 / 1e6, "output": 10.00 / 1e6},
    "gemini-2.5-flash": {"input": 0.15 / 1e6, "output": 0.60 / 1e6},
    # DeepSeek
    "deepseek-chat": {"input": 0.27 / 1e6, "output": 1.10 / 1e6},
    "deepseek-reasoner": {"input": 0.55 / 1e6, "output": 2.19 / 1e6},
    # Groq-hosted models
    "llama-3.3-70b-versatile": {"input": 0.59 / 1e6, "output": 0.79 / 1e6},
    "mixtral-8x7b-32768": {"input": 0.24 / 1e6, "output": 0.24 / 1e6},
}


@dataclass
class _Record:
    input_tokens: int
    output_tokens: int
    latency_ms: float


@dataclass
class _QualityRecord:
    call_id: int
    faithfulness: float | None
    relevancy: float | None
    timestamp: float

    @property
    def mean_score(self) -> float | None:
        vals = [v for v in (self.faithfulness, self.relevancy) if v is not None]
        if not vals:
            return None
        return sum(vals) / len(vals)


class TokenTracer:
    """Track token usage, latency, estimated cost, and quality trends per session."""

    def __init__(
        self,
        model: str,
        enabled: bool = True,
        quality_window: int = 5,
        quality_trend_threshold: float = 0.03,
    ) -> None:
        self.model = model
        self.enabled = enabled
        self.quality_window = max(2, quality_window)
        self.quality_trend_threshold = max(0.0, quality_trend_threshold)
        self._records: list[_Record] = []
        self._quality_records: list[_QualityRecord] = []

    def record(
        self,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
    ) -> int:
        """Record one LLM call and return its 1-based call id."""
        if not self.enabled:
            return 0
        self._records.append(_Record(input_tokens, output_tokens, latency_ms))
        return len(self._records)

    def record_quality(
        self,
        faithfulness: float | None = None,
        relevancy: float | None = None,
        call_id: int | None = None,
        timestamp: float | None = None,
    ) -> int:
        """Record quality metrics for a call (faithfulness/relevancy)."""
        if not self.enabled:
            return 0
        if faithfulness is None and relevancy is None:
            return 0

        resolved_call_id = call_id if call_id is not None else len(self._records)
        self._quality_records.append(
            _QualityRecord(
                call_id=max(0, resolved_call_id),
                faithfulness=faithfulness,
                relevancy=relevancy,
                timestamp=timestamp if timestamp is not None else time.time(),
            )
        )
        return resolved_call_id

    def quality_history(self) -> list[dict[str, float | int | None]]:
        """Return recorded quality metrics in insertion order."""
        return [
            {
                "call_id": r.call_id,
                "faithfulness": r.faithfulness,
                "relevancy": r.relevancy,
                "timestamp": r.timestamp,
            }
            for r in self._quality_records
        ]

    def _quality_trend(self) -> str:
        scores = [r.mean_score for r in self._quality_records]
        numeric_scores = [s for s in scores if s is not None]

        if len(numeric_scores) < self.quality_window:
            return "stable"

        recent = numeric_scores[-self.quality_window :]
        split = len(recent) // 2
        if split == 0:
            return "stable"

        first = recent[:split]
        second = recent[split:]
        first_avg = sum(first) / len(first)
        second_avg = sum(second) / len(second)
        delta = second_avg - first_avg

        if delta >= self.quality_trend_threshold:
            return "improving"
        if delta <= -self.quality_trend_threshold:
            return "degrading"
        return "stable"

    def summary(self) -> dict:
        total_input = sum(r.input_tokens for r in self._records)
        total_output = sum(r.output_tokens for r in self._records)
        total_latency = sum(r.latency_ms for r in self._records)

        costs = COST_TABLE.get(self.model, {})
        cost_input = total_input * costs.get("input", 0.0)
        cost_output = total_output * costs.get("output", 0.0)

        faithfulness_values = [
            r.faithfulness for r in self._quality_records if r.faithfulness is not None
        ]
        relevancy_values = [r.relevancy for r in self._quality_records if r.relevancy is not None]

        avg_faithfulness = (
            sum(faithfulness_values) / len(faithfulness_values) if faithfulness_values else None
        )
        avg_relevancy = sum(relevancy_values) / len(relevancy_values) if relevancy_values else None

        return {
            "model": self.model,
            "calls": len(self._records),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_latency_ms": round(total_latency, 2),
            "estimated_cost_usd": round(cost_input + cost_output, 6),
            "avg_faithfulness": round(avg_faithfulness, 4)
            if avg_faithfulness is not None
            else None,
            "avg_relevancy": round(avg_relevancy, 4) if avg_relevancy is not None else None,
            "quality_trend": self._quality_trend(),
        }

    def reset(self) -> None:
        self._records.clear()
        self._quality_records.clear()

    def start_timer(self) -> float:
        """Return current time in ms for use with record()."""
        return time.monotonic() * 1000

    def elapsed_ms(self, start: float) -> float:
        return time.monotonic() * 1000 - start
