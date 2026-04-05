from __future__ import annotations

import json
from typing import Any

from .base import BaseSplitter


class JSONSplitter(BaseSplitter):
    """
    Split JSON arrays or objects into manageable chunks.

    For JSON arrays: each element is a split candidate.
    For JSON objects: each top-level key/value pair is a split candidate.
    Candidates are grouped into chunks up to chunk_size characters.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 0,
    ) -> None:
        """
        Initialize the JSONSplitter.

        Args:
            chunk_size: Maximum size in characters for each chunk.
            chunk_overlap: Number of characters to overlap between chunks.

        Raises:
            ValueError: If chunk_size is not positive or chunk_overlap is negative.
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
        Split JSON text into chunks.

        Args:
            text: JSON string (array or object).

        Returns:
            List of JSON string chunks.

        Raises:
            ValueError: If input is not valid JSON.
        """
        text = text.strip()
        if not text:
            return []

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON input: {e}") from e

        # Convert to list of candidates
        candidates = self._extract_candidates(data)

        if not candidates:
            return []

        # Convert candidates back to JSON strings
        candidate_strs = [json.dumps(c, ensure_ascii=False) for c in candidates]

        # Group into chunks
        return self._create_chunks(candidate_strs)

    def _extract_candidates(self, data: Any) -> list[Any]:
        """
        Extract split candidates from JSON data.

        Args:
            data: Parsed JSON data (list or dict).

        Returns:
            List of split candidates.
        """
        if isinstance(data, list):
            # Each array element is a candidate
            return data
        elif isinstance(data, dict):
            # Each key/value pair is a candidate
            return [{k: v} for k, v in data.items()]
        else:
            # Primitive value — return as single candidate
            return [data]

    def _create_chunks(self, candidates: list[str]) -> list[str]:
        """
        Group candidate strings into chunks respecting chunk_size.

        Args:
            candidates: List of JSON strings to chunk.

        Returns:
            List of chunked JSON strings (as arrays).
        """
        if not candidates:
            return []

        chunks: list[str] = []
        current_group: list[str] = []
        current_size = 0

        # Account for array brackets and commas
        array_overhead = 2  # "[]"

        for candidate in candidates:
            candidate_size = len(candidate)

            # Calculate size with this candidate
            # Size = brackets + candidates + commas between them
            comma_overhead = len(current_group)  # One comma per existing element
            projected_size = (
                array_overhead + current_size + candidate_size + comma_overhead
            )

            if projected_size <= self.chunk_size:
                # Add to current group
                current_group.append(candidate)
                current_size += candidate_size
            else:
                # Current group is full, save it
                if current_group:
                    chunks.append(self._group_to_json(current_group))

                # Start new group
                if candidate_size + array_overhead <= self.chunk_size:
                    current_group = [candidate]
                    current_size = candidate_size
                else:
                    # Single candidate exceeds chunk_size — add it anyway
                    chunks.append(self._group_to_json([candidate]))
                    current_group = []
                    current_size = 0

        # Add remaining group
        if current_group:
            chunks.append(self._group_to_json(current_group))

        # Apply overlap if needed
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks)

        return chunks

    def _group_to_json(self, group: list[str]) -> str:
        """
        Convert a group of JSON strings into a single JSON array string.

        Args:
            group: List of JSON strings.

        Returns:
            JSON array string containing the group.
        """
        # Parse each candidate back to object
        objects = [json.loads(s) for s in group]
        # Return as JSON array
        return json.dumps(objects, ensure_ascii=False)

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """
        Apply overlap between chunks by including trailing elements from previous chunk.

        Args:
            chunks: List of chunk strings.

        Returns:
            List of overlapped chunk strings.
        """
        if len(chunks) < 2:
            return chunks

        overlapped: list[str] = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = json.loads(chunks[i - 1])
            curr_chunk = json.loads(chunks[i])

            # Calculate how many characters we can take from previous chunk
            overlap_chars = 0
            overlap_elements: list[Any] = []

            # Take elements from the end of previous chunk
            for elem in reversed(prev_chunk):
                elem_str = json.dumps(elem, ensure_ascii=False)
                if overlap_chars + len(elem_str) <= self.chunk_overlap:
                    overlap_elements.insert(0, elem)
                    overlap_chars += len(elem_str)
                else:
                    break

            # Merge overlap with current chunk
            if overlap_elements:
                merged = overlap_elements + curr_chunk
                overlapped.append(json.dumps(merged, ensure_ascii=False))
            else:
                overlapped.append(chunks[i])

        return overlapped
