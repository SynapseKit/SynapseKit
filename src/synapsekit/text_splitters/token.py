from __future__ import annotations

from .base import BaseSplitter
from .recursive import RecursiveCharacterTextSplitter

# Rough estimate: 1 token ~ 4 characters for English text
_CHARS_PER_TOKEN = 4


class TokenAwareSplitter(BaseSplitter):
    """
    Split text so each chunk fits within a token budget.

    Uses a simple heuristic of ~4 characters per token and delegates
    to ``RecursiveCharacterTextSplitter`` for the actual splitting.
    """

    def __init__(
        self,
        max_tokens: int = 256,
        chunk_overlap: int = 50,
        chars_per_token: int = _CHARS_PER_TOKEN,
    ) -> None:
        self.max_tokens = max_tokens
        self.chunk_overlap = chunk_overlap
        self.chars_per_token = chars_per_token
        self._inner = RecursiveCharacterTextSplitter(
            chunk_size=max_tokens * chars_per_token,
            chunk_overlap=chunk_overlap,
        )

    def split(self, text: str) -> list[str]:
        return self._inner.split(text)
