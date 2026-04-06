from __future__ import annotations

import json

from .base import BaseSplitter


class JSONSplitter(BaseSplitter):
    """Split JSON arrays or objects into manageable chunks."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e

        # Extract split candidates
        if isinstance(data, list):
            candidates = [json.dumps(item, ensure_ascii=False) for item in data]
        elif isinstance(data, dict):
            candidates = [
                json.dumps({key: value}, ensure_ascii=False) for key, value in data.items()
            ]
        else:
            # Primitive value (string, number, boolean, null)
            candidates = [json.dumps(data, ensure_ascii=False)]

        if not candidates:
            return []

        return self._merge(candidates)

    def _merge(self, parts: list[str]) -> list[str]:
        chunks: list[str] = []
        current = ""

        for part in parts:
            # Try to add part to current chunk
            if current:
                # Join with newline for readability
                candidate = f"{current}\n{part}"
            else:
                candidate = part

            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            # Current chunk is full, save it
            if current:
                chunks.append(current)

            # Handle oversized single part
            if len(part) > self.chunk_size:
                # Hard split oversized part
                step = (
                    self.chunk_size - self.chunk_overlap
                    if self.chunk_overlap > 0
                    else self.chunk_size
                )
                for i in range(0, len(part), step):
                    chunks.append(part[i : i + self.chunk_size])
                current = ""
            else:
                current = part

        if current:
            chunks.append(current)

        # Apply overlap if requested
        if self.chunk_overlap <= 0 or len(chunks) < 2:
            return chunks

        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-self.chunk_overlap :]
            overlapped.append(tail + chunks[i])
        return overlapped
