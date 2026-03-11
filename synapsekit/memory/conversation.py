from __future__ import annotations

from typing import List


class ConversationMemory:
    """
    Sliding-window conversation history.
    Keeps the last `window` turns (user + assistant pairs).
    """

    def __init__(self, window: int = 10) -> None:
        self._window = window
        self._messages: List[dict] = []

    def add(self, role: str, content: str) -> None:
        """Append a message. Role should be 'user' or 'assistant'."""
        self._messages.append({"role": role, "content": content})
        # Keep sliding window: window * 2 messages (user + assistant each turn)
        max_messages = self._window * 2
        if len(self._messages) > max_messages:
            self._messages = self._messages[-max_messages:]

    def get_messages(self) -> List[dict]:
        """Return the current message history."""
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def format_context(self) -> str:
        """Flatten history to a plain string for prompt injection."""
        parts = []
        for m in self._messages:
            role = m["role"].capitalize()
            parts.append(f"{role}: {m['content']}")
        return "\n".join(parts)

    def __len__(self) -> int:
        return len(self._messages)
