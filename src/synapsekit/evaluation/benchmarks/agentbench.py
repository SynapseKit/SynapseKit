"""AgentBench benchmark integration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from .base import BaseBenchmark, BenchmarkResult


class AgentBenchBenchmark(BaseBenchmark):
    """AgentBench evaluation suite."""

    name: ClassVar[str] = "AgentBench"

    def load_dataset(self, split: str = "test") -> list[dict[str, Any]]:
        """Load the AgentBench dataset from HuggingFace."""
        try:
            import datasets
        except ImportError as e:
            raise ImportError("The 'datasets' package is required for AgentBench. Run `pip install datasets`.") from e

        ds = datasets.load_dataset("THUDM/AgentBench", split=split)
        return [dict(row) for row in ds]

    def evaluate(
        self,
        agent: Callable[[dict[str, Any]], Any],
        split: str = "test",
        limit: int | None = None,
    ) -> BenchmarkResult:
        """Run the AgentBench evaluation."""
        dataset = self.load_dataset(split)
        if limit is not None:
            dataset = dataset[:limit]

        total = len(dataset)
        success = 0
        errors = []

        for task in dataset:
            try:
                result = agent(task)

                # Simple exact match fallback; actual AgentBench is multi-turn interaction
                expected = str(task.get("reference", "")).strip().casefold()
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
