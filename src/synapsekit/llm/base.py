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
    # Caching
    cache: bool = False
    cache_maxsize: int = 128
    # Retries
    max_retries: int = 0
    retry_delay: float = 1.0


class BaseLLM(ABC):
    """Abstract base for all LLM providers."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._cache = None
        if config.cache:
            from ._cache import AsyncLRUCache

            self._cache = AsyncLRUCache(maxsize=config.cache_maxsize)

    @abstractmethod
    async def stream(self, prompt: str, **kw) -> AsyncGenerator[str]:
        """Yield text tokens as they arrive."""
        ...

    async def generate(self, prompt: str, **kw) -> str:
        """Collect all streamed tokens into a single string."""
        cache_key = None
        if self._cache is not None:
            from ._cache import AsyncLRUCache

            cache_key = AsyncLRUCache.make_key(
                self.config.model,
                prompt,
                kw.get("temperature", self.config.temperature),
                kw.get("max_tokens", self.config.max_tokens),
            )
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        result = await self._generate_with_retry(prompt, **kw)

        if self._cache is not None and cache_key is not None:
            self._cache.put(cache_key, result)
        return result

    async def _generate_with_retry(self, prompt: str, **kw) -> str:
        """Generate with optional retry logic."""
        if self.config.max_retries > 0:
            from ._retry import retry_async

            return await retry_async(
                self._generate_uncached,
                prompt,
                max_retries=self.config.max_retries,
                delay=self.config.retry_delay,
                **kw,
            )
        return await self._generate_uncached(prompt, **kw)

    async def _generate_uncached(self, prompt: str, **kw) -> str:
        """Raw generate — no cache, no retry."""
        return "".join([t async for t in self.stream(prompt, **kw)])

    async def stream_with_messages(self, messages: list[dict], **kw) -> AsyncGenerator[str]:
        """Stream from a messages list (role/content dicts)."""
        prompt = _messages_to_prompt(messages)
        async for token in self.stream(prompt, **kw):
            yield token

    async def generate_with_messages(self, messages: list[dict], **kw) -> str:
        """Generate from a messages list."""
        cache_key = None
        if self._cache is not None:
            from ._cache import AsyncLRUCache

            cache_key = AsyncLRUCache.make_key(
                self.config.model,
                messages,
                kw.get("temperature", self.config.temperature),
                kw.get("max_tokens", self.config.max_tokens),
            )
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        if self.config.max_retries > 0:
            from ._retry import retry_async

            result = await retry_async(
                self._generate_with_messages_uncached,
                messages,
                max_retries=self.config.max_retries,
                delay=self.config.retry_delay,
                **kw,
            )
        else:
            result = await self._generate_with_messages_uncached(messages, **kw)

        if self._cache is not None and cache_key is not None:
            self._cache.put(cache_key, result)
        return result

    async def _generate_with_messages_uncached(self, messages: list[dict], **kw) -> str:
        """Raw generate_with_messages — no cache, no retry."""
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
