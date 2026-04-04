from __future__ import annotations

import re

from .base import BaseSplitter


class SentenceTextSplitter(BaseSplitter):
    """Split text into chunks by grouping complete sentences."""

    def __init__(
        self,
        chunk_size: int = 10,
        chunk_overlap: int = 1,
    ) -> None:
        """
        Initialize the SentenceTextSplitter.

        Args:
            chunk_size: Number of sentences per chunk.
            chunk_overlap: Number of sentences to overlap between chunks.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[str]:
        """
        Split text into chunks by grouping complete sentences.

        Args:
            text: The text to split.

        Returns:
            List of text chunks, where each chunk contains chunk_size sentences.
        """
        text = text.strip()
        if not text:
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        if len(sentences) <= self.chunk_size:
            return [" ".join(sentences)]

        return self._create_chunks(sentences)

    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences using regex to detect sentence boundaries.

        Args:
            text: The text to split.

        Returns:
            List of sentences.
        """
        # Split on sentence boundaries: . ! ?
        # Keep the punctuation with the sentence
        # Handle edge cases like "Dr." "Mr." "U.S.A."
        pattern = r"(?<=[.!?])\s+"
        sentences = re.split(pattern, text)

        # Clean up and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def _create_chunks(self, sentences: list[str]) -> list[str]:
        """
        Create overlapping chunks from a list of sentences.

        Args:
            sentences: List of sentences to chunk.

        Returns:
            List of text chunks with overlap.
        """
        chunks: list[str] = []
        i = 0

        while i < len(sentences):
            # Get chunk_size sentences starting from i
            end = min(i + self.chunk_size, len(sentences))
            chunk_sentences = sentences[i:end]
            chunks.append(" ".join(chunk_sentences))

            # Move forward by (chunk_size - chunk_overlap) sentences
            # This creates the overlap
            if self.chunk_overlap > 0:
                i += self.chunk_size - self.chunk_overlap
            else:
                i += self.chunk_size

            # Avoid infinite loop if we're at the end
            if i >= len(sentences):
                break

        return chunks
