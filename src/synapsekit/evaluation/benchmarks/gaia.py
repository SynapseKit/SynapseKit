"""GAIA benchmark integration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from .base import BaseBenchmark, BenchmarkResult


class GAIABenchmark(BaseBenchmark):
    """GAIA (General AI Assistant) Benchmark suite."""

    name: ClassVar[str] = "GAIA"

    def load_dataset(self, split: str = "validation") -> list[dict[str, Any]]:
        """Load the GAIA dataset.

        Loads raw task records; some GAIA tasks also require fetching
        attached files, which isn't handled here.
        """
        try:
            from datasets import load_dataset
        except ImportError as err:
            raise ImportError(
                "The datasets library is required to load GAIA. "
                "Install it with `pip install synapsekit[huggingface]`."
            ) from err

        dataset = load_dataset("gaia-benchmark/GAIA", "2023_all", split=split)
        return list(dataset)

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
                result = agent(task)
                expected = task.get("expected_answer")
                if expected:
                    # When ground truth is available, require a real match
                    # rather than just any non-null response.
                    is_success = isinstance(result, str) and expected in result
                else:
                    # Placeholder grading: no expected answer to check against.
                    is_success = result is not False and result is not None
                if is_success:
                    success += 1
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
