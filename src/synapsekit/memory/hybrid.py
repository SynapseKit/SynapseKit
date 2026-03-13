"""Hybrid conversation memory: sliding window + LLM summary of older messages."""

from __future__ import annotations

from typing import Any


class HybridMemory:
    """Combined sliding-window + summary conversation memory.

    Keeps recent messages in full, and summarizes older messages using an LLM.
    This gives the model both precise recent context and compressed long-term context.

    Usage::

        from synapsekit.memory.hybrid import HybridMemory

        memory = HybridMemory(llm=llm, window=5, summary_max_tokens=200)
        memory.add("user", "Hello!")
        memory.add("assistant", "Hi there!")
        # ... many more messages ...

        messages = await memory.get_messages_with_summary()
        # Returns: [{"role": "system", "content": "Summary: ..."}, ...recent messages...]
    """

    def __init__(
        self,
        llm: Any,
        window: int = 5,
        summary_max_tokens: int = 200,
    ) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self._llm = llm
        self._window = window
        self._summary_max_tokens = summary_max_tokens
        self._messages: list[dict] = []
        self._summary: str = ""

    def add(self, role: str, content: str) -> None:
        """Append a message."""
        self._messages.append({"role": role, "content": content})

    def get_messages(self) -> list[dict]:
        """Return all messages (no summary applied)."""
        return list(self._messages)

    def get_recent_messages(self) -> list[dict]:
        """Return only the recent window of messages."""
        max_messages = self._window * 2
        return list(self._messages[-max_messages:])

    async def _summarize_messages(self, messages: list[dict]) -> str:
        """Use the LLM to summarize a list of messages."""
        if not messages:
            return ""

        conversation = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)
        prompt = (
            f"Summarize this conversation in {self._summary_max_tokens} tokens or less. "
            f"Capture the key points, decisions, and context:\n\n{conversation}"
        )
        return await self._llm.generate(prompt)

    async def get_messages_with_summary(self) -> list[dict]:
        """Return messages with older ones replaced by a summary.

        If there are more messages than the window allows, the older
        messages are summarized and prepended as a system message.
        """
        max_messages = self._window * 2

        if len(self._messages) <= max_messages:
            return list(self._messages)

        # Split into old (to summarize) and recent (to keep)
        old_messages = self._messages[:-max_messages]
        recent_messages = self._messages[-max_messages:]

        # Summarize old messages
        self._summary = await self._summarize_messages(old_messages)

        result: list[dict] = []
        if self._summary:
            result.append(
                {
                    "role": "system",
                    "content": f"Summary of earlier conversation:\n{self._summary}",
                }
            )
        result.extend(recent_messages)
        return result

    async def format_context(self) -> str:
        """Format the conversation with summary for prompt injection."""
        messages = await self.get_messages_with_summary()
        parts = []
        for m in messages:
            role = m["role"].capitalize()
            parts.append(f"{role}: {m['content']}")
        return "\n".join(parts)

    @property
    def summary(self) -> str:
        """The current summary of older messages (empty until first summarization)."""
        return self._summary

    def clear(self) -> None:
        """Clear all messages and summary."""
        self._messages.clear()
        self._summary = ""

    def __len__(self) -> int:
        return len(self._messages)
