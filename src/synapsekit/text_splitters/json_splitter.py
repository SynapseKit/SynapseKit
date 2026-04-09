"""JSON-aware text splitter."""

from __future__ import annotations

import json
from typing import Any

from .base import BaseSplitter


class JSONSplitter(BaseSplitter):
    """Split JSON arrays or objects into manageable chunks.

    - If input is a JSON **array**: each element is a split candidate.
    - If input is a JSON **object**: each top-level key/value pair is a split candidate.
    - Candidates are greedily grouped into chunks up to *chunk_size* characters.
    - Each output chunk is a valid JSON string.

    Example::

        splitter = JSONSplitter(chunk_size=500)
        chunks = splitter.split('[{"id":1},{"id":2},{"id":3}]')
        # Each chunk is a valid JSON array string containing as many elements as fit.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 0,
        ensure_ascii: bool = False,
    ) -> None:
        """
        Initialize the JSONSplitter.

        Args:
            chunk_size: Maximum chunk size in characters.
            chunk_overlap: Minimum number of characters worth of items to repeat
                at the start of each successive chunk. Overlap is applied at the
                item level so every output chunk remains valid JSON.
            ensure_ascii: If True, escape non-ASCII characters in output.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ensure_ascii = ensure_ascii

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(self, text: str) -> list[str]:
        """Split a JSON string into a list of valid JSON chunk strings.

        Args:
            text: A valid JSON string (array, object, or scalar).

        Returns:
            A list of JSON strings, each at most *chunk_size* characters
            (except for oversized single elements that undergo a hard split).

        Raises:
            ValueError: If *text* is not valid JSON.
        """
        text = text.strip()
        if not text:
            return []

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Input is not valid JSON: {exc}") from exc

        # --- Scalar (string, number, bool, null) --------------------------
        if not isinstance(data, (list, dict)):
            return [text]

        # --- Array --------------------------------------------------------
        if isinstance(data, list):
            candidates = [self._serialize(item) for item in data]
            return self._group_and_wrap(candidates, wrapper="array")

        # --- Object -------------------------------------------------------
        candidates = [self._serialize({k: v}) for k, v in data.items()]
        return self._group_and_wrap(candidates, wrapper="object")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _serialize(self, obj: Any) -> str:
        """Serialize a Python object to a compact JSON string."""
        return json.dumps(obj, ensure_ascii=self.ensure_ascii, separators=(",", ":"))

    def _group_and_wrap(
        self,
        candidates: list[str],
        wrapper: str,
    ) -> list[str]:
        """Greedily group serialized candidates into chunks.

        Each chunk is wrapped as either a JSON array ``[…]`` or a merged
        JSON object ``{…}`` depending on *wrapper*.

        When ``chunk_overlap > 0``, the last N characters worth of *items*
        from chunk K are prepended to chunk K+1 so that every output chunk
        remains valid JSON.
        """
        if not candidates:
            return []

        chunks: list[str] = []
        # Each entry is the list of serialized items that went into that chunk.
        chunk_item_groups: list[list[str]] = []
        current_items: list[str] = []
        # Overhead: "[" + "]" or "{" + "}" = 2 chars, plus commas between items
        current_length = 2  # opening + closing bracket

        for candidate in candidates:
            # Cost of adding this candidate: the candidate itself + a comma if
            # it's not the first item in the current group.
            comma_cost = 1 if current_items else 0
            addition_cost = len(candidate) + comma_cost

            if current_length + addition_cost <= self.chunk_size:
                current_items.append(candidate)
                current_length += addition_cost
            else:
                # Flush current group
                if current_items:
                    chunks.append(self._wrap(current_items, wrapper))
                    chunk_item_groups.append(current_items)

                # Check if the single candidate itself exceeds chunk_size
                if len(candidate) + 2 > self.chunk_size:
                    # Hard-split the oversized candidate as plain text
                    chunks.extend(self._hard_split(candidate))
                    chunk_item_groups.extend([[c] for c in self._hard_split(candidate)])
                    current_items = []
                    current_length = 2
                else:
                    current_items = [candidate]
                    current_length = 2 + len(candidate)

        # Flush remaining
        if current_items:
            chunks.append(self._wrap(current_items, wrapper))
            chunk_item_groups.append(current_items)

        if self.chunk_overlap <= 0 or len(chunks) < 2:
            return chunks

        return self._apply_item_overlap(chunk_item_groups, wrapper)

    @staticmethod
    def _wrap(items: list[str], wrapper: str) -> str:
        """Wrap a list of serialized items into a JSON array or merged object."""
        if wrapper == "array":
            return "[" + ",".join(items) + "]"

        # Object: each item is like '{"key":value}' — strip outer braces and merge
        inner_parts: list[str] = []
        for item in items:
            # Remove leading '{' and trailing '}'
            inner_parts.append(item[1:-1])
        return "{" + ",".join(inner_parts) + "}"

    def _hard_split(self, text: str) -> list[str]:
        """Fall-back character-level split for an oversized single element."""
        step = self.chunk_size - self.chunk_overlap if self.chunk_overlap > 0 else self.chunk_size
        return [text[i : i + self.chunk_size] for i in range(0, len(text), step)]

    def _apply_item_overlap(
        self,
        chunk_item_groups: list[list[str]],
        wrapper: str,
    ) -> list[str]:
        """Re-wrap chunks so that the tail items of chunk N appear at the
        start of chunk N+1, keeping every output chunk valid JSON.

        Items are greedily added from the tail of the previous group until
        the cumulative character count of the overlap items reaches or
        exceeds ``chunk_overlap``.
        """
        result = [self._wrap(chunk_item_groups[0], wrapper)]
        for i in range(1, len(chunk_item_groups)):
            prev_items = chunk_item_groups[i - 1]
            # Collect tail items from prev group until overlap budget is met
            overlap_items: list[str] = []
            total = 0
            for item in reversed(prev_items):
                total += len(item) + (1 if overlap_items else 0)
                overlap_items.insert(0, item)
                if total >= self.chunk_overlap:
                    break
            combined = overlap_items + chunk_item_groups[i]
            result.append(self._wrap(combined, wrapper))
        return result
