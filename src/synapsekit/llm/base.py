import asyncio
import inspect
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMConfig:
    model: str
    api_key: str
    provider: str  # "openai" | "anthropic"
    system_prompt: str = "You are a helpful assistant."
    temperature: float = 0.2
    max_tokens: int = 1024
    top_p: float | None = None
    # Caching
    cache: bool = False
    cache_maxsize: int = 128
    cache_backend: str = "memory"  # "memory", "sqlite", "filesystem", or "redis"
    cache_db_path: str = "synapsekit_llm_cache.db"
    # Retries
    max_retries: int = 2
    retry_delay: float = 1.0
    # Per-request timeout (seconds) passed to providers that build their own
    # HTTP client. ``None`` means "use the provider/SDK default".
    timeout: float | None = None
    # Rate limiting
    requests_per_minute: int | None = None

    def __post_init__(self) -> None:
        """Validate config values eagerly so bad settings fail at construction.

        Intentionally conservative — ``api_key`` is *not* required, since many
        local providers (Ollama, llama.cpp, LM Studio, MLX, ...) need none.
        """
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError(f"temperature must be between 0.0 and 2.0, got {self.temperature!r}")
        if self.top_p is not None and not (0.0 <= self.top_p <= 1.0):
            raise ValueError(f"top_p must be between 0.0 and 1.0, got {self.top_p!r}")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be greater than 0, got {self.max_tokens!r}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries!r}")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError(f"timeout must be greater than 0, got {self.timeout!r}")


class BaseLLM(ABC):
    """Abstract base for all LLM providers."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._cache: Any = None
        self._rate_limiter: Any = None
        if config.cache:
            if config.cache_backend == "sqlite":
                from ._sqlite_cache import SQLiteLLMCache

                self._cache = SQLiteLLMCache(db_path=config.cache_db_path)
            elif config.cache_backend == "filesystem":
                from ._filesystem_cache import FilesystemLLMCache

                self._cache = FilesystemLLMCache(cache_dir=config.cache_db_path)
            elif config.cache_backend == "redis":
                from ._redis_cache import RedisLLMCache

                self._cache = RedisLLMCache(url=config.cache_db_path)
            else:
                from ._cache import AsyncLRUCache

                self._cache = AsyncLRUCache(maxsize=config.cache_maxsize)
        if config.requests_per_minute is not None:
            from ._rate_limit import TokenBucketRateLimiter

            self._rate_limiter = TokenBucketRateLimiter(config.requests_per_minute)

    @abstractmethod
    async def stream(self, prompt: str, **kw: Any) -> AsyncGenerator[str]:
        """Yield text tokens as they arrive."""
        yield ""  # pragma: no cover

    def _make_cache_key(self, prompt_or_messages: str | list[dict[str, Any]], **kw: Any) -> str:
        """Build a cache key that also isolates by system prompt and provider.

        ``AsyncLRUCache.make_key`` intentionally keeps its 4-argument signature
        (shared with the Rust fast path), so ``system_prompt`` and ``provider``
        are folded into the ``model`` component to keep entries from two
        instances that share a cache backend but differ in system prompt /
        provider from colliding.
        """
        from ._cache import AsyncLRUCache

        model_scope = (
            f"{self.config.provider}\x1f{self.config.system_prompt}\x1f{self.config.model}"
        )
        return AsyncLRUCache.make_key(
            model_scope,
            prompt_or_messages,
            kw.get("temperature", self.config.temperature),
            kw.get("max_tokens", self.config.max_tokens),
        )

    async def _cache_get(self, key: str) -> Any | None:
        """Read from the cache without blocking the event loop.

        Persistent backends (SQLite/Redis) use blocking sync I/O, so run them in
        the default executor. The in-memory LRU is cheap enough to call inline.
        """
        if self._cache is None:
            return None
        from ._cache import AsyncLRUCache

        if isinstance(self._cache, AsyncLRUCache):
            return self._cache.get(key)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._cache.get, key)

    async def _cache_put(self, key: str, value: Any) -> None:
        """Write to the cache without blocking the event loop."""
        if self._cache is None:
            return
        from ._cache import AsyncLRUCache

        if isinstance(self._cache, AsyncLRUCache):
            self._cache.put(key, value)
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._cache.put, key, value)

    async def generate(self, prompt: str, **kw: Any) -> str:
        """Collect all streamed tokens into a single string."""
        cache_key: str | None = None
        if self._cache is not None:
            cache_key = self._make_cache_key(prompt, **kw)
            cached = await self._cache_get(cache_key)
            if cached is not None:
                return str(cached)

        result = await self._generate_with_retry(prompt, **kw)

        if self._cache is not None and cache_key is not None:
            await self._cache_put(cache_key, result)
        return result

    async def _generate_with_retry(self, prompt: str, **kw: Any) -> str:
        """Generate with optional retry logic."""
        if self.config.max_retries > 0:
            from ._retry import retry_async

            return str(
                await retry_async(
                    self._generate_uncached,
                    prompt,
                    max_retries=self.config.max_retries,
                    delay=self.config.retry_delay,
                    **kw,
                )
            )
        return await self._generate_uncached(prompt, **kw)

    async def _acquire_rate_limit(self) -> None:
        """Wait for rate limiter if configured."""
        if self._rate_limiter is not None:
            await self._rate_limiter.acquire()

    async def _generate_uncached(self, prompt: str, **kw: Any) -> str:
        """Raw generate — no cache, no retry."""
        await self._acquire_rate_limit()
        return "".join([t async for t in self.stream(prompt, **kw)])

    async def stream_with_messages(
        self, messages: list[dict[str, Any]], **kw: Any
    ) -> AsyncGenerator[str]:
        """Stream from a messages list (role/content dicts)."""
        prompt = _messages_to_prompt(messages)
        async for token in self.stream(prompt, **kw):
            yield token

    async def generate_with_messages(self, messages: list[dict[str, Any]], **kw: Any) -> str:
        """Generate from a messages list."""
        cache_key: str | None = None
        if self._cache is not None:
            cache_key = self._make_cache_key(messages, **kw)
            cached = await self._cache_get(cache_key)
            if cached is not None:
                return str(cached)

        if self.config.max_retries > 0:
            from ._retry import retry_async

            result = str(
                await retry_async(
                    self._generate_with_messages_uncached,
                    messages,
                    max_retries=self.config.max_retries,
                    delay=self.config.retry_delay,
                    **kw,
                )
            )
        else:
            result = await self._generate_with_messages_uncached(messages, **kw)

        if self._cache is not None and cache_key is not None:
            await self._cache_put(cache_key, result)
        return result

    async def _generate_with_messages_uncached(
        self, messages: list[dict[str, Any]], **kw: Any
    ) -> str:
        """Raw generate_with_messages — no cache, no retry."""
        await self._acquire_rate_limit()
        return "".join([t async for t in self.stream_with_messages(messages, **kw)])

    async def call_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Native function-calling with optional retry and rate limiting.

        Override ``_call_with_tools_impl`` in providers that support it.
        """
        await self._acquire_rate_limit()
        if self.config.max_retries > 0:
            from ._retry import retry_async

            return dict(
                await retry_async(
                    self._call_with_tools_impl,
                    messages,
                    tools,
                    max_retries=self.config.max_retries,
                    delay=self.config.retry_delay,
                )
            )
        return await self._call_with_tools_impl(messages, tools)

    async def _call_with_tools_impl(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Override in providers that support native function calling."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support native function calling."
        )

    @property
    def cache_stats(self) -> dict[str, int]:
        """Return cache hit/miss/size statistics, or empty dict if caching is disabled."""
        if self._cache is None:
            return {}
        return {
            "hits": self._cache.hits,
            "misses": self._cache.misses,
            "size": len(self._cache),
        }

    @property
    def tokens_used(self) -> dict[str, int]:
        return {"input": self._input_tokens, "output": self._output_tokens}

    def _reset_tokens(self) -> None:
        self._input_tokens = 0
        self._output_tokens = 0

    async def aclose(self) -> None:
        """Close a lazily-created async provider client, when present."""
        client = getattr(self, "_client", None)
        if client is None:
            return
        close = getattr(client, "aclose", None) or getattr(client, "close", None)
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result
        self._client = None

    async def __aenter__(self) -> "BaseLLM":
        return self

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        await self.aclose()


def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    """Fallback: flatten messages list to a plain prompt string."""
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"{role.capitalize()}: {content}")
    return "\n".join(parts)
