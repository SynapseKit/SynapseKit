from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseCheckpointer(ABC):
    """Abstract base for graph state checkpointers."""

    @abstractmethod
    def save(self, graph_id: str, step: int, state: dict[str, Any]) -> None:
        """Persist the state at the given step."""
        ...

    @abstractmethod
    def load(self, graph_id: str) -> tuple[int, dict[str, Any]] | None:
        """Load the most recent checkpoint. Returns ``(step, state)`` or ``None``."""
        ...

    @abstractmethod
    def delete(self, graph_id: str) -> None:
        """Remove the checkpoint for the given graph_id."""
        ...
