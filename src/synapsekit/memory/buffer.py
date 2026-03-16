from __future__ import annotations


class BufferMemory:
    """Unbounded conversation buffer — keeps all messages until cleared.

    The simplest memory backend: no windowing, no trimming, no LLM calls.

    Usage::

        mem = BufferMemory()
        mem.add("user", "Hello!")
        mem.add("assistant", "Hi there!")
        print(mem.format_context())
    """

    def __init__(self) -> None:
        self._messages: list[dict] = []

    def add(self, role: str, content: str) -> None:
        """Append a message."""
        self._messages.append({"role": role, "content": content})

    def get_messages(self) -> list[dict]:
        """Return a copy of all messages."""
        return list(self._messages)

    def format_context(self) -> str:
        """Flatten history to a plain string for prompt injection."""
        parts = []
        for m in self._messages:
            role = m["role"].capitalize()
            parts.append(f"{role}: {m['content']}")
        return "\n".join(parts)

    def clear(self) -> None:
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)
