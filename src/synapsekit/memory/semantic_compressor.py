"""Semantic Compressor Memory: Dynamic context compression for long-running agents."""

from __future__ import annotations

from typing import Any


class SemanticCompressorMemory:
    """An advanced memory backend that automatically summarizes older parts of a conversation.

    Instead of simply dropping old messages (which causes 'amnesia') or maintaining a
    naive sliding window, this memory dynamically compresses history into dense summaries
    using a provided 'compressor' LLM when the context window reaches a token threshold.

    It maintains a running structured summary of "Core Context" and "Recent Events"
    to ensure the agent never loses the core context while staying well within token limits.
    """

    def __init__(
        self,
        llm: Any,
        max_tokens: int = 4000,
        compression_threshold: float = 0.8,
        chars_per_token: int = 4,
    ) -> None:
        """Initialize the Semantic Compressor.

        Args:
            llm: A BaseLLM instance (preferably a small, fast model like gpt-4o-mini)
            max_tokens: The absolute maximum token limit for the context window.
            compression_threshold: When buffer exceeds (max_tokens * compression_threshold), compression triggers.
            chars_per_token: Approximate characters per token for estimation.
        """
        if max_tokens < 500:
            raise ValueError("max_tokens must be >= 500")

        self._llm = llm
        self._max_tokens = max_tokens
        self._compression_threshold = compression_threshold
        self._chars_per_token = chars_per_token

        self._messages: list[dict] = []
        self._summary: str = ""

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // self._chars_per_token

    def _buffer_tokens(self) -> int:
        total = sum(self._estimate_tokens(m["content"]) for m in self._messages)
        if self._summary:
            total += self._estimate_tokens(self._summary)
        return total

    def add(self, role: str, content: str) -> None:
        """Append a message to the conversation."""
        self._messages.append({"role": role, "content": content})

    async def get_messages(self) -> list[dict]:
        """Return messages, triggering semantic compression if threshold is exceeded."""
        limit = int(self._max_tokens * self._compression_threshold)

        # Trigger compression if we exceed the threshold and have enough messages to compress
        while self._buffer_tokens() > limit and len(self._messages) > 4:
            # We compress the oldest half of the current messages
            split_idx = len(self._messages) // 2
            to_compress = self._messages[:split_idx]

            conversation = "\n".join(
                f"{m['role'].capitalize()}: {m['content']}" for m in to_compress
            )

            if self._summary:
                prompt = (
                    "You are a memory compression engine for an AI agent. "
                    "Your task is to merge the 'Existing Memory' with the 'New Exchanges' into a single, "
                    "dense, updated summary. Retain core instructions, critical entities, and ongoing state.\n\n"
                    f"### Existing Memory:\n{self._summary}\n\n"
                    f"### New Exchanges to Compress:\n{conversation}\n\n"
                    "### Updated Compressed Memory:"
                )
            else:
                prompt = (
                    "You are a memory compression engine for an AI agent. "
                    "Summarize the following conversation into a dense block of context. "
                    "Retain core user instructions, critical entities, and ongoing state.\n\n"
                    f"### Conversation to Compress:\n{conversation}\n\n"
                    "### Compressed Memory:"
                )

            # Generate the new compressed memory
            new_summary = await self._llm.generate(prompt)
            self._summary = new_summary.strip()

            # Prune the compressed messages
            self._messages = self._messages[split_idx:]

        # Reconstruct the payload
        result: list[dict] = []
        if self._summary:
            result.append(
                {
                    "role": "system",
                    "content": f"[SYSTEM: The following is a highly compressed semantic memory of earlier events]\n\n{self._summary}",
                }
            )
        result.extend(self._messages)
        return result

    def format_context(self) -> str:
        """Flatten current buffer to a plain string."""
        parts = []
        if self._summary:
            parts.append(f"Compressed Memory:\n{self._summary}")
        for m in self._messages:
            role = m["role"].capitalize()
            parts.append(f"{role}: {m['content']}")
        return "\n\n".join(parts)

    @property
    def summary(self) -> str:
        """The current semantic summary of older events."""
        return self._summary

    def clear(self) -> None:
        """Clear all messages and the compressed memory."""
        self._messages.clear()
        self._summary = ""

    def __len__(self) -> int:
        return len(self._messages)
