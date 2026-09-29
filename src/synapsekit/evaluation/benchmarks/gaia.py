"""GAIA benchmark integration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from .base import BaseBenchmark, BenchmarkResult


class GAIABenchmark(BaseBenchmark):
    """GAIA (General AI Assistant) Benchmark suite."""

    name: ClassVar[str] = "GAIA"

    def load_dataset(self, split: str = "validation") -> list[dict[str, Any]]:
        """Load the GAIA dataset using the datasets library."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' package is required to load the GAIA dataset. "
                "Please install it with `pip install datasets`."
            )
            
        dataset = load_dataset("gaia-benchmark/GAIA", "2023_all", split=split)
        return [dict(item) for item in dataset]

    def evaluate(
        self,
        agent: Callable[[dict[str, Any]], Any],
        split: str = "validation",
        limit: int | None = None,
    ) -> BenchmarkResult:
        """Run the GAIA evaluation."""
        dataset = self.load_dataset(split)
        if limit is not None:
            dataset = dataset[:limit]

        total = len(dataset)
        success = 0
        errors = []

        for task in dataset:
            try:
                agent(task)
                # TODO: compare agent output against task["expected_answer"] and increment success
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
