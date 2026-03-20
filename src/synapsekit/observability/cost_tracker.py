"""Hierarchical cost attribution with scope context manager."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .tracer import COST_TABLE


@dataclass
class CostRecord:
    """A single cost record from an LLM call."""

    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    scope_path: str
    timestamp: float = field(default_factory=time.time)


class _ScopeContext:
    """Manages the scope stack for hierarchical cost attribution."""

    def __init__(self, tracker: CostTracker, name: str) -> None:
        self._tracker = tracker
        self._name = name

    def __enter__(self) -> _ScopeContext:
        self._tracker._push_scope(self._name)
        return self

    def __exit__(self, *_: Any) -> None:
        self._tracker._pop_scope()


class CostTracker:
    """Hierarchical cost attribution with scope context manager.

    Usage::

        tracker = CostTracker()
        with tracker.scope("pipeline"):
            with tracker.scope("retrieval"):
                tracker.record("gpt-4o-mini", 500, 200, 120.0)
            with tracker.scope("generation"):
                tracker.record("gpt-4o", 1000, 500, 350.0)

        print(tracker.total_cost_usd)
        print(tracker.summary())
    """

    def __init__(self) -> None:
        self._records: list[CostRecord] = []
        self._scope_stack: list[str] = []
        self._lock = threading.Lock()

    def scope(self, name: str) -> _ScopeContext:
        """Return a context manager that pushes/pops a named scope."""
        return _ScopeContext(self, name)

    def _push_scope(self, name: str) -> None:
        with self._lock:
            self._scope_stack.append(name)

    def _pop_scope(self) -> None:
        with self._lock:
            if self._scope_stack:
                self._scope_stack.pop()

    @property
    def _current_scope_path(self) -> str:
        return "/".join(self._scope_stack) if self._scope_stack else "(root)"

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
    ) -> CostRecord:
        """Record an LLM call with auto-calculated cost from COST_TABLE."""
        pricing = COST_TABLE.get(model)
        if pricing:
            cost = input_tokens * pricing["input"] + output_tokens * pricing["output"]
        else:
            cost = 0.0

        with self._lock:
            rec = CostRecord(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                latency_ms=latency_ms,
                scope_path=self._current_scope_path,
            )
            self._records.append(rec)
        return rec

    @property
    def total_cost_usd(self) -> float:
        """Total cost across all records."""
        with self._lock:
            return sum(r.cost_usd for r in self._records)

    @property
    def records(self) -> list[CostRecord]:
        """All recorded cost records."""
        with self._lock:
            return list(self._records)

    def summary(self) -> dict[str, Any]:
        """Nested dict grouped by scope path.

        Returns a dict like::

            {
                "pipeline/retrieval": {"total_cost_usd": 0.00025, "calls": 1, ...},
                "pipeline/generation": {"total_cost_usd": 0.0075, "calls": 1, ...},
            }
        """
        with self._lock:
            scopes: dict[str, dict[str, Any]] = {}
            for rec in self._records:
                if rec.scope_path not in scopes:
                    scopes[rec.scope_path] = {
                        "total_cost_usd": 0.0,
                        "total_input_tokens": 0,
                        "total_output_tokens": 0,
                        "calls": 0,
                        "total_latency_ms": 0.0,
                    }
                s = scopes[rec.scope_path]
                s["total_cost_usd"] += rec.cost_usd
                s["total_input_tokens"] += rec.input_tokens
                s["total_output_tokens"] += rec.output_tokens
                s["calls"] += 1
                s["total_latency_ms"] += rec.latency_ms
            return scopes

    def reset(self) -> None:
        """Clear all recorded data."""
        with self._lock:
            self._records.clear()
            self._scope_stack.clear()
