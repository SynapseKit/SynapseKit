"""Token Buffer Memory: drop oldest messages when buffer exceeds a token limit."""

from __future__ import annotations


class TokenBufferMemory:
    """Conversation memory that drops oldest messages when the buffer
    exceeds an approximate token limit.

    Unlike :class:`SummaryBufferMemory`, this does **not** use an LLM to
    summarise — it simply discards the oldest messages.

    Usage::

        memory = TokenBufferMemory(max_tokens=4000)
        memory.add("user", "Hello!")
        memory.add("assistant", "Hi there!")
        messages = memory.get_messages()
    """

    def __init__(
        self,
        max_tokens: int = 4000,
        chars_per_token: int = 4,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if chars_per_token < 1:
            raise ValueError("chars_per_token must be >= 1")
        self._max_tokens = max_tokens
        self._chars_per_token = chars_per_token
        self._messages: list[dict] = []

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from character length."""
        return len(text) // self._chars_per_token

    def _total_tokens(self) -> int:
        """Estimate total tokens across all buffered messages."""
        return sum(self._estimate_tokens(m["content"]) for m in self._messages)

    def _trim(self) -> None:
        """Drop oldest messages until the buffer fits within the token budget."""
        while self._total_tokens() > self._max_tokens and self._messages:
            self._messages.pop(0)

    def add(self, role: str, content: str) -> None:
        """Append a message and trim if over budget."""
        self._messages.append({"role": role, "content": content})
        self._trim()

    def get_messages(self) -> list[dict]:
        """Return the current message history."""
        return list(self._messages)

    def format_context(self) -> str:
        """Flatten history to a plain string for prompt injection."""
        parts = []
        for m in self._messages:
            role = m["role"].capitalize()
            parts.append(f"{role}: {m['content']}")
        return "\n".join(parts)

    def clear(self) -> None:
        """Clear all messages."""
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)
