"""AgentBench benchmark integration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from .base import BaseBenchmark, BenchmarkResult


class AgentBenchBenchmark(BaseBenchmark):
    """AgentBench evaluation suite."""

    name: ClassVar[str] = "AgentBench"

    def load_dataset(self, split: str = "test") -> list[dict[str, Any]]:
        """Load the AgentBench dataset.

        Currently a stub implementation.
        """
        try:
            from datasets import load_dataset
        except ImportError as err:
            raise ImportError(
                "The datasets library is required to load AgentBench. "
                "Install it with `pip install synapsekit[huggingface]`."
            ) from err

        dataset = load_dataset("THUDM/AgentBench", "os", split=split)
        return list(dataset)

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
                if result is not False and result is not None:
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
