from __future__ import annotations

from .base import BaseSplitter


class CharacterTextSplitter(BaseSplitter):
    """Split text on a single separator string."""

    def __init__(
        self,
        separator: str = "\n\n",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> None:
        self.separator = separator
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        parts = text.split(self.separator)
        if len(parts) <= 1:
            # Hard split as last resort
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
            ]
        return self._merge(parts)

    def _merge(self, parts: list[str]) -> list[str]:
        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = current + (self.separator if current else "") + part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > self.chunk_size:
                    # Hard split oversized single part
                    for i in range(0, len(part), self.chunk_size - self.chunk_overlap):
                        chunks.append(part[i : i + self.chunk_size])
                    current = ""
                else:
                    current = part
        if current:
            chunks.append(current)

        # Apply overlap
        if self.chunk_overlap <= 0 or len(chunks) < 2:
            return chunks

        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-self.chunk_overlap :]
            overlapped.append(tail + chunks[i])
        return overlapped
