"""Summary Buffer Memory: summarize when buffer exceeds a token limit."""

from __future__ import annotations

from typing import Any


class SummaryBufferMemory:
    """Conversation memory that summarizes older messages when the buffer
    exceeds a token limit.

    Unlike ``HybridMemory`` (fixed window), this tracks approximate token
    count and summarizes only when needed.

    Usage::

        memory = SummaryBufferMemory(llm=llm, max_tokens=2000)
        memory.add("user", "Hello!")
        memory.add("assistant", "Hi there!")
        messages = await memory.get_messages()
    """

    def __init__(
        self,
        llm: Any,
        max_tokens: int = 2000,
        chars_per_token: int = 4,
    ) -> None:
        if max_tokens < 100:
            raise ValueError("max_tokens must be >= 100")
        self._llm = llm
        self._max_tokens = max_tokens
        self._chars_per_token = chars_per_token
        self._messages: list[dict] = []
        self._summary: str = ""

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from character length."""
        return len(text) // self._chars_per_token

    def _buffer_tokens(self) -> int:
        """Estimate total tokens in the current buffer."""
        total = sum(self._estimate_tokens(m["content"]) for m in self._messages)
        if self._summary:
            total += self._estimate_tokens(self._summary)
        return total

    def add(self, role: str, content: str) -> None:
        """Append a message."""
        self._messages.append({"role": role, "content": content})

    async def get_messages(self) -> list[dict]:
        """Return messages, summarizing older ones if buffer exceeds limit."""
        if self._buffer_tokens() <= self._max_tokens:
            result: list[dict] = []
            if self._summary:
                result.append(
                    {
                        "role": "system",
                        "content": f"Summary of earlier conversation:\n{self._summary}",
                    }
                )
            result.extend(self._messages)
            return result

        # Summarize oldest messages until we're under the limit
        while self._buffer_tokens() > self._max_tokens and len(self._messages) > 2:
            # Take the oldest pair of messages to summarize
            to_summarize = self._messages[:2]

            conversation = "\n".join(
                f"{m['role'].capitalize()}: {m['content']}" for m in to_summarize
            )
            if self._summary:
                prompt = (
                    f"Update this conversation summary with the new exchanges. "
                    f"Keep it concise.\n\n"
                    f"Current summary: {self._summary}\n\n"
                    f"New exchanges:\n{conversation}"
                )
            else:
                prompt = f"Summarize this conversation exchange concisely:\n\n{conversation}"

            # Await the LLM first; only mutate state after the summary is
            # successfully computed. If generate() raises, self._summary and
            # self._messages are left untouched so the caller can retry.
            new_summary = await self._llm.generate(prompt)
            self._summary = new_summary
            self._messages = self._messages[2:]

        result = []
        if self._summary:
            result.append(
                {
                    "role": "system",
                    "content": f"Summary of earlier conversation:\n{self._summary}",
                }
            )
        result.extend(self._messages)
        return result

    def format_context(self) -> str:
        """Flatten current buffer to a plain string (sync, no summarization)."""
        parts = []
        if self._summary:
            parts.append(f"Summary: {self._summary}")
        for m in self._messages:
            role = m["role"].capitalize()
            parts.append(f"{role}: {m['content']}")
        return "\n".join(parts)

    @property
    def summary(self) -> str:
        """The current running summary."""
        return self._summary

    def clear(self) -> None:
        """Clear all messages and summary."""
        self._messages.clear()
        self._summary = ""

    def __len__(self) -> int:
        return len(self._messages)
