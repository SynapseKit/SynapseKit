from __future__ import annotations

from .base import BaseSplitter


class RecursiveCharacterTextSplitter(BaseSplitter):
    """
    Recursive character text splitter.

    Tries splitting by paragraphs, then sentences, then words, then hard split.
    This is the same algorithm previously embedded in ``RAGPipeline``.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " "]

    def split(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        for sep in self.separators:
            parts = text.split(sep)
            if len(parts) > 1:
                return self._merge(parts, sep)

        # Hard split as last resort
        return [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
        ]

    def _merge(self, parts: list[str], sep: str) -> list[str]:
        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > self.chunk_size:
                    chunks.extend(self.split(part))
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
