from __future__ import annotations

import time
from dataclasses import dataclass

# Cost per token in USD (input, output)
COST_TABLE: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15 / 1e6, "output": 0.60 / 1e6},
    "gpt-4o": {"input": 2.50 / 1e6, "output": 10.00 / 1e6},
    "gpt-4o-2024-11-20": {"input": 2.50 / 1e6, "output": 10.00 / 1e6},
    "gpt-4-turbo": {"input": 10.00 / 1e6, "output": 30.00 / 1e6},
    "claude-haiku-4-5-20251001": {"input": 0.80 / 1e6, "output": 4.00 / 1e6},
    "claude-sonnet-4-6": {"input": 3.00 / 1e6, "output": 15.00 / 1e6},
    "claude-opus-4-6": {"input": 15.00 / 1e6, "output": 75.00 / 1e6},
}


@dataclass
class _Record:
    input_tokens: int
    output_tokens: int
    latency_ms: float


class TokenTracer:
    """Track token usage, latency, and estimated cost per session."""

    def __init__(self, model: str, enabled: bool = True) -> None:
        self.model = model
        self.enabled = enabled
        self._records: list[_Record] = []

    def record(
        self,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
    ) -> None:
        if not self.enabled:
            return
        self._records.append(_Record(input_tokens, output_tokens, latency_ms))

    def summary(self) -> dict:
        total_input = sum(r.input_tokens for r in self._records)
        total_output = sum(r.output_tokens for r in self._records)
        total_latency = sum(r.latency_ms for r in self._records)

        costs = COST_TABLE.get(self.model, {})
        cost_input = total_input * costs.get("input", 0.0)
        cost_output = total_output * costs.get("output", 0.0)

        return {
            "model": self.model,
            "calls": len(self._records),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_latency_ms": round(total_latency, 2),
            "estimated_cost_usd": round(cost_input + cost_output, 6),
        }

    def reset(self) -> None:
        self._records.clear()

    def start_timer(self) -> float:
        """Return current time in ms for use with record()."""
        return time.monotonic() * 1000

    def elapsed_ms(self, start: float) -> float:
        return time.monotonic() * 1000 - start
