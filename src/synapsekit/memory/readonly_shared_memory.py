"""ReadOnlySharedMemory: read-only view wrapper over any memory object."""

from __future__ import annotations

from typing import Any


class ReadOnlySharedMemory:
    """Read-only view over any existing memory object.

    Wraps any memory that has a ``messages`` property or ``get_context()``
    method. ``add_message()`` raises ``PermissionError``.

    Useful for sharing an agent's memory context with subordinate agents
    without letting them write to it.

    Usage::

        shared = ReadOnlySharedMemory(memory=agent_memory)
        context = await shared.get_context("user query")
        shared.add_message("user", "msg")  # raises PermissionError
    """

    def __init__(self, memory: Any) -> None:
        self._memory = memory

    @property
    def messages(self) -> list[dict]:
        """Proxy the wrapped memory's messages property."""
        return list(self._memory.messages)

    async def get_context(self, query: str | None = None) -> list[dict]:
        """Proxy the wrapped memory's get_context method (if available)."""
        if hasattr(self._memory, "get_context"):
            if query is not None:
                result = self._memory.get_context(query)
            else:
                result = self._memory.get_context()
            # Support both coroutines and plain values
            if hasattr(result, "__await__"):
                return await result
            return result
        # Fallback: return messages list
        return list(self._memory.messages)

    def add_message(self, role: str, content: str) -> None:
        """Raise PermissionError — this memory is read-only."""
        raise PermissionError("This memory is read-only")
