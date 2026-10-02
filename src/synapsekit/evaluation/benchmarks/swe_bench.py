"""SWE-bench benchmark integration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from .base import BaseBenchmark, BenchmarkResult


class SWEBenchmark(BaseBenchmark):
    """SWE-bench evaluation suite."""

    name: ClassVar[str] = "SWE-bench"

    def load_dataset(self, split: str = "test") -> list[dict[str, Any]]:
        """Load the SWE-bench dataset from HuggingFace."""
        try:
            import datasets
        except ImportError as e:
            raise ImportError("The 'datasets' package is required for SWE-bench. Run `pip install datasets`.") from e

        ds = datasets.load_dataset("princeton-nlp/SWE-bench", split=split)
        return [dict(row) for row in ds]

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
                patch = agent(task)

                # In SWE-bench, verifying a patch strictly requires executing it against the repo's test suite.
                # Since isolated test execution requires the heavy `swebench` docker harness,
                # we do a lightweight offline fallback check here for perfect semantic matches
                # against the gold patch (or we assume failure if not identical/highly similar).
                # Note: This is a fast-path approximation for testing.
                gold_patch = str(task.get("patch", "")).strip()
                if patch and gold_patch and str(patch).strip() == gold_patch:
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
