"""SWE-bench benchmark integration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from .base import BaseBenchmark, BenchmarkResult


class SWEBenchmark(BaseBenchmark):
    """SWE-bench evaluation suite."""

    name: ClassVar[str] = "SWE-bench"

    def load_dataset(self, split: str = "test") -> list[dict[str, Any]]:
        """Load the SWE-bench dataset using the datasets library."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' package is required to load the SWE-bench dataset. "
                "Please install it with `pip install datasets`."
            )
            
        dataset = load_dataset("princeton-nlp/SWE-bench", split=split)
        return [dict(item) for item in dataset]

    def evaluate(
        self,
        agent: Callable[[dict[str, Any]], Any],
        split: str = "test",
        limit: int | None = None,
    ) -> BenchmarkResult:
        """Run the SWE-bench evaluation."""
        dataset = self.load_dataset(split)
        if limit is not None:
            dataset = dataset[:limit]

        total = len(dataset)
        success = 0
        errors = []

        for task in dataset:
            try:
                agent(task)
                # TODO: run patch against test suite and increment success if all tests pass
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
