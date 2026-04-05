from __future__ import annotations

import re
from typing import Any

from .base import BaseSplitter


class SentenceWindowSplitter(BaseSplitter):
    """
    Split text into individual sentences with surrounding context windows.

    Each chunk contains a target sentence plus a configurable number of
    surrounding sentences before and after. Useful for retrieval systems
    where you want to store the target sentence but embed it with context.
    """

    def __init__(self, window_size: int = 2) -> None:
        """
        Initialize the SentenceWindowSplitter.

        Args:
            window_size: Number of sentences to include before and after
                        the target sentence. Default is 2.

        Raises:
            ValueError: If window_size is negative.
        """
        if window_size < 0:
            raise ValueError("window_size cannot be negative")

        self.window_size = window_size

    def split(self, text: str) -> list[str]:
        """
        Split text into sentence windows.

        Args:
            text: The text to split.

        Returns:
            List of text chunks, where each chunk contains a target sentence
            with surrounding context sentences.
        """
        text = text.strip()
        if not text:
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        if len(sentences) == 1:
            return [sentences[0]]

        return self._create_windows(sentences)

    def split_with_metadata(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Split text into sentence windows with metadata.

        Each chunk includes:
        - "text": The full window (target sentence + context)
        - "metadata": Contains "target_sentence", "chunk_index", and parent metadata

        Args:
            text: The text to split.
            metadata: Optional metadata to attach to each chunk.

        Returns:
            List of dicts with text and metadata for each sentence window.
        """
        text = text.strip()
        if not text:
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        if len(sentences) == 1:
            base_metadata = metadata.copy() if metadata else {}
            base_metadata["chunk_index"] = 0
            base_metadata["target_sentence"] = sentences[0]
            return [{"text": sentences[0], "metadata": base_metadata}]

        result: list[dict[str, Any]] = []
        base_metadata = metadata.copy() if metadata else {}

        for idx in range(len(sentences)):
            start = max(0, idx - self.window_size)
            end = min(len(sentences), idx + self.window_size + 1)

            window_text = " ".join(sentences[start:end])

            chunk_metadata = base_metadata.copy()
            chunk_metadata["chunk_index"] = idx
            chunk_metadata["target_sentence"] = sentences[idx]

            result.append({"text": window_text, "metadata": chunk_metadata})

        return result

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences on . ! ? boundaries."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _create_windows(self, sentences: list[str]) -> list[str]:
        """Build one window per sentence, including up to window_size neighbours."""
        windows: list[str] = []
        for idx in range(len(sentences)):
            start = max(0, idx - self.window_size)
            end = min(len(sentences), idx + self.window_size + 1)
            windows.append(" ".join(sentences[start:end]))
        return windows
