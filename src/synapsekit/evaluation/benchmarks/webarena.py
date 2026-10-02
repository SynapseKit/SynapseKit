"""WebArena benchmark integration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from .base import BaseBenchmark, BenchmarkResult


class WebArenaBenchmark(BaseBenchmark):
    """WebArena evaluation suite."""

    name: ClassVar[str] = "WebArena"

    def load_dataset(self, split: str = "test") -> list[dict[str, Any]]:
        """Load the WebArena dataset from HuggingFace."""
        try:
            import datasets
        except ImportError as e:
            raise ImportError("The 'datasets' package is required for WebArena. Run `pip install datasets`.") from e

        ds = datasets.load_dataset("jykoh/webarena", split=split)
        return [dict(row) for row in ds]

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
                result = agent(task)

                # WebArena evaluation requires scraping a live environment and comparing DOM state.
                # As a fallback proxy, we check if the agent's text output contains the target text.
                expected = str(task.get("eval", {}).get("reference_answers", [""])[0]).strip().casefold()
                prediction = str(result).strip().casefold() if result is not None else ""

                if prediction and expected and (expected == prediction or expected in prediction):
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
