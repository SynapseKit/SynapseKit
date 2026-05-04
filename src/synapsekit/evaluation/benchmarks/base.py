"""Base classes for agent benchmarking suites."""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass
class BenchmarkResult:
    """Result of a benchmark evaluation."""

    benchmark_name: str
    total_tasks: int
    successful_tasks: int
    score: float
    details: dict[str, Any]

    def format_leaderboard(self) -> str:
        """Format the result as a simple leaderboard/report string."""
        lines = [
            f"=== {self.benchmark_name} Leaderboard ===",
            f"Score: {self.score:.4f}",
            f"Tasks: {self.successful_tasks}/{self.total_tasks} successful",
        ]
        if self.details:
            lines.append("Details:")
            for k, v in self.details.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


class BaseBenchmark(abc.ABC):
    """Abstract base class for all agent benchmarks."""

    #: Human-readable name for the benchmark. Subclasses must set this as a class variable.
    name: ClassVar[str]

    @abc.abstractmethod
    def load_dataset(self, split: str = "test") -> Any:
        """Load the benchmark dataset."""
        pass

    @abc.abstractmethod
    def evaluate(
        self, agent: Callable[[dict[str, Any]], Any], split: str = "test", limit: int | None = None
    ) -> BenchmarkResult:
        """Evaluate an agent against the benchmark.

        Args:
            agent: A callable representing the agent to be evaluated. It should take a task dictionary and return an output.
            split: The dataset split to evaluate on.
            limit: Maximum number of tasks to evaluate.

        Returns:
            BenchmarkResult: The result of the evaluation.
        """
        pass
