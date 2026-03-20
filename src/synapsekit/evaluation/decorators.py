"""``@eval_case`` decorator for defining evaluation test cases."""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalCaseMeta:
    """Metadata attached to an ``@eval_case``-decorated function."""

    min_score: float | None = None
    max_cost_usd: float | None = None
    max_latency_ms: float | None = None
    tags: list[str] = field(default_factory=list)


def eval_case(
    *,
    min_score: float | None = None,
    max_cost_usd: float | None = None,
    max_latency_ms: float | None = None,
    tags: list[str] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to mark a function as an evaluation case.

    The decorated function remains a normal callable (sync or async),
    compatible with pytest and ``synapsekit test``.

    Usage::

        @eval_case(min_score=0.8, max_cost_usd=0.05, tags=["rag", "qa"])
        async def test_summarization():
            ...
            return {"score": 0.85, "cost_usd": 0.02, "latency_ms": 1200}
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        meta = EvalCaseMeta(
            min_score=min_score,
            max_cost_usd=max_cost_usd,
            max_latency_ms=max_latency_ms,
            tags=tags or [],
        )
        fn._eval_case_meta = meta  # type: ignore[attr-defined]

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

        wrapper._eval_case_meta = meta  # type: ignore[attr-defined]
        return wrapper

    return decorator
