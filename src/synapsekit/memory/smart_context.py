"""Smart Context Manager: hierarchical allocation, summarization, and prompt caching."""

from __future__ import annotations

from typing import Any


class SmartContextManager:
    """
    Intelligent context window management with hierarchical allocation,
    sliding window pruning, smart summarization, and prompt caching tags.

    Hierarchy:
    1. System Prompt (static, cached)
    2. Summary (updated, cached)
    3. Search Results (dynamic, bounded)
    4. Recent Messages (sliding window)

    This integration achieves substantial cost savings (e.g., up to 80% reduction)
    by leveraging prompt caching on the system and summary blocks with expensive models,
    while using a cheaper model to compress the rolling window.

    Usage::

        manager = SmartContextManager(
            cheap_llm=cheap_llm,
            max_recent_tokens=4000,
            max_search_tokens=2000
        )
        manager.set_system("You are a helpful assistant.")
        manager.add_search_results("Document A...")
        manager.add("user", "Hello!")
        messages = await manager.get_messages()
    """

    def __init__(
        self,
        cheap_llm: Any = None,
        max_recent_tokens: int = 4000,
        max_search_tokens: int = 2000,
        chars_per_token: int = 4,
    ) -> None:
        self._cheap_llm = cheap_llm
        self._max_recent_tokens = max_recent_tokens
        self._max_search_tokens = max_search_tokens
        self._chars_per_token = chars_per_token

        self._system: str = ""
        self._summary: str = ""
        self._search_results: str = ""
        self._messages: list[dict] = []

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from character length."""
        return len(text) // self._chars_per_token

    def _recent_tokens(self) -> int:
        """Estimate total tokens in the recent messages buffer."""
        return sum(self._estimate_tokens(m["content"]) for m in self._messages)

    def set_system(self, content: str) -> None:
        """Set the static system prompt."""
        self._system = content

    def set_search_results(self, content: str) -> None:
        """Set dynamic search results context."""
        self._search_results = content

    def clear_search_results(self) -> None:
        """Clear dynamic search results."""
        self._search_results = ""

    def add(self, role: str, content: str) -> None:
        """Append a new message to the recent buffer."""
        self._messages.append({"role": role, "content": content})

    async def get_messages(self) -> list[dict]:
        """
        Return the hierarchical message list, summarizing older messages
        if the recent buffer exceeds its token limit. Injects cache_control
        metadata on the system and summary blocks.
        """
        # Summarize older messages if over budget
        while self._recent_tokens() > self._max_recent_tokens and len(self._messages) > 2:
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

            if self._cheap_llm:
                # Await the LLM first; only mutate state after the summary is successfully computed.
                new_summary = await self._cheap_llm.generate(prompt)
                self._summary = new_summary
            else:
                # If no LLM is provided for summarization, we simply drop the messages.
                pass

            # Prune the summarized/dropped messages
            self._messages = self._messages[2:]

        result = []

        # 1. System Prompt (Cached via ephemeral tag)
        if self._system:
            result.append(
                {
                    "role": "system",
                    "content": self._system,
                    "cache_control": {"type": "ephemeral"},
                }
            )

        # 2. Summary (Cached via ephemeral tag)
        if self._summary:
            result.append(
                {
                    "role": "system",
                    "content": f"Summary of earlier conversation:\n{self._summary}",
                    "cache_control": {"type": "ephemeral"},
                }
            )

        # 3. Search Results (Dynamic, not cached, bounded)
        if self._search_results:
            allowed_chars = self._max_search_tokens * self._chars_per_token
            # Safe simple truncation for token limits
            trimmed_search = self._search_results[:allowed_chars]
            result.append(
                {
                    "role": "system",
                    "content": f"Search Results Context:\n{trimmed_search}",
                }
            )

        # 4. Recent Messages
        result.extend(self._messages)
        return result

    @property
    def summary(self) -> str:
        """The current running summary."""
        return self._summary

    def clear(self) -> None:
        """Clear all context and messages."""
        self._messages.clear()
        self._summary = ""
        self._system = ""
        self._search_results = ""

    def __len__(self) -> int:
        return len(self._messages)
