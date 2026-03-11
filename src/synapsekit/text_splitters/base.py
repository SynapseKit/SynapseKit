from __future__ import annotations

from abc import ABC, abstractmethod


class BaseSplitter(ABC):
    """Abstract base for all text splitters."""

    @abstractmethod
    def split(self, text: str) -> list[str]:
        """Split *text* into a list of chunks."""
        ...
