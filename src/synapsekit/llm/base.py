from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass


@dataclass
class LLMConfig:
    model: str
    api_key: str
    provider: str  # "openai" | "anthropic"
    system_prompt: str = "You are a helpful assistant."
    temperature: float = 0.2
    max_tokens: int = 1024


class BaseLLM(ABC):
    """Abstract base for all LLM providers."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._input_tokens: int = 0
        self._output_tokens: int = 0

    @abstractmethod
    async def stream(self, prompt: str, **kw) -> AsyncGenerator[str]:
        """Yield text tokens as they arrive."""
        ...

    async def generate(self, prompt: str, **kw) -> str:
        """Collect all streamed tokens into a single string."""
        return "".join([t async for t in self.stream(prompt, **kw)])

    async def stream_with_messages(self, messages: list[dict], **kw) -> AsyncGenerator[str]:
        """Stream from a messages list (role/content dicts)."""
        prompt = _messages_to_prompt(messages)
        async for token in self.stream(prompt, **kw):
            yield token

    async def generate_with_messages(self, messages: list[dict], **kw) -> str:
        """Generate from a messages list."""
        return "".join([t async for t in self.stream_with_messages(messages, **kw)])

    @property
    def tokens_used(self) -> dict:
        return {"input": self._input_tokens, "output": self._output_tokens}

    def _reset_tokens(self) -> None:
        self._input_tokens = 0
        self._output_tokens = 0


def _messages_to_prompt(messages: list[dict]) -> str:
    """Fallback: flatten messages list to a plain prompt string."""
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"{role.capitalize()}: {content}")
    return "\n".join(parts)
