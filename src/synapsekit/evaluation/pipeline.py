"""Evaluation pipeline — run multiple metrics on RAG outputs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from .base import MetricResult


@dataclass
class EvaluationResult:
    """Aggregated evaluation result."""

    scores: dict[str, float] = field(default_factory=dict)
    details: dict[str, MetricResult] = field(default_factory=dict)

    @property
    def mean_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)

    def __repr__(self) -> str:
        scores_str = ", ".join(f"{k}={v:.2f}" for k, v in self.scores.items())
        return f"EvaluationResult(mean={self.mean_score:.2f}, {scores_str})"


class EvaluationPipeline:
    """Run multiple evaluation metrics on RAG outputs.

    All metrics for a single sample are evaluated **concurrently** via
    ``asyncio.gather``, so total latency equals the slowest metric rather
    than the sum of all metric latencies.

    Batch evaluation also runs samples concurrently, gated by a configurable
    semaphore to avoid overloading downstream LLM APIs.

    Usage::
        pipeline = EvaluationPipeline(metrics=[
            FaithfulnessMetric(llm),
            RelevancyMetric(llm),
            GroundednessMetric(llm),
        ])
        result = await pipeline.evaluate(
            question="What is Python?",
            answer="Python is a programming language.",
            contexts=["Python is a high-level programming language."],
        )
        print(result.mean_score)
        print(result.scores)
    """

    def __init__(self, metrics: list[Any]) -> None:
        self._metrics = metrics

    async def evaluate(
        self,
        question: str = "",
        answer: str = "",
        contexts: list[str] | None = None,
    ) -> EvaluationResult:
        # All metrics evaluated concurrently — latency = max(metric latencies)
        metric_results: list[MetricResult] = await asyncio.gather(
            *[
                metric.evaluate(
                    question=question,
                    answer=answer,
                    contexts=contexts or [],
                )
                for metric in self._metrics
            ]
        )

        scores = {m.name: r.score for m, r in zip(self._metrics, metric_results)}
        details = {m.name: r for m, r in zip(self._metrics, metric_results)}
        return EvaluationResult(scores=scores, details=details)

    async def evaluate_batch(
        self,
        samples: list[dict[str, Any]],
        concurrency: int = 10,
    ) -> list[EvaluationResult]:
        """Evaluate a batch of samples concurrently.

        Args:
            samples: List of dicts with keys ``question``, ``answer``,
                ``contexts`` passed as kwargs to :meth:`evaluate`.
            concurrency: Maximum number of samples evaluated simultaneously.
                Defaults to 10 to avoid overwhelming upstream LLM APIs.
        """
        sem = asyncio.Semaphore(concurrency)

        async def _eval(sample: dict[str, Any]) -> EvaluationResult:
            async with sem:
                return await self.evaluate(**sample)

        return list(await asyncio.gather(*[_eval(s) for s in samples]))
