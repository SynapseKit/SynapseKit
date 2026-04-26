"""WebArena benchmark integration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from .base import BaseBenchmark, BenchmarkResult


class WebArenaBenchmark(BaseBenchmark):
    """WebArena evaluation suite."""

    name: ClassVar[str] = "WebArena"

    def load_dataset(self, split: str = "test") -> list[dict[str, Any]]:
        """Load the WebArena dataset.

        Currently a stub implementation.
        """
        # TODO: load tasks from the WebArena task JSON files
        return []

    def evaluate(
        self,
        agent: Callable[[dict[str, Any]], Any],
        split: str = "test",
        limit: int | None = None,
    ) -> BenchmarkResult:
        """Run the WebArena evaluation."""
        dataset = self.load_dataset(split)
        if limit is not None:
            dataset = dataset[:limit]

        total = len(dataset)
        success = 0
        errors = []

        for task in dataset:
            try:
                agent(task)
                # TODO: verify task completion via WebArena evaluator and increment success
            except Exception as e:
                errors.append(str(e))

        score = success / total if total > 0 else 0.0

        return BenchmarkResult(
            benchmark_name=self.name,
            total_tasks=total,
            successful_tasks=success,
            score=score,
            details={"split": split, "errors": errors},
        )
