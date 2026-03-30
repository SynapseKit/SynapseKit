from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseSplitter(ABC):
    """Abstract base for all text splitters."""

    @abstractmethod
    def split(self, text: str) -> list[str]:
        """Split *text* into a list of chunks."""
        ...

    def split_with_metadata(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Split *text* into chunks, preserving metadata for each chunk.

        Args:
            text: The text to split.
            metadata: Optional metadata to attach to each chunk.

        Returns:
            List of dicts with keys:
                - "text": The chunk text
                - "metadata": Merged metadata including parent metadata and chunk_index
        """
        chunks = self.split(text)
        result: list[dict[str, Any]] = []

        base_metadata = metadata.copy() if metadata else {}

        for idx, chunk_text in enumerate(chunks):
            chunk_metadata = base_metadata.copy()
            chunk_metadata["chunk_index"] = idx
            result.append({"text": chunk_text, "metadata": chunk_metadata})

        return result
