"""Tests for LLM response caching and exponential backoff retries."""

import pytest

from synapsekit.llm._cache import AsyncLRUCache
from synapsekit.llm._retry import retry_async
from synapsekit.llm.base import BaseLLM, LLMConfig

# ------------------------------------------------------------------ #
# AsyncLRUCache
# ------------------------------------------------------------------ #


class TestAsyncLRUCache:
    def test_put_and_get(self):
        cache = AsyncLRUCache(maxsize=2)
        cache.put("a", "hello")
        assert cache.get("a") == "hello"

    def test_miss_returns_none(self):
        cache = AsyncLRUCache(maxsize=2)
        assert cache.get("missing") is None

    def test_eviction(self):
        cache = AsyncLRUCache(maxsize=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)  # should evict "a"
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_lru_order(self):
        cache = AsyncLRUCache(maxsize=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")  # Touch "a" — now "b" is LRU
        cache.put("c", 3)  # should evict "b"
        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3

    def test_clear(self):
        cache = AsyncLRUCache(maxsize=10)
        cache.put("a", 1)
        cache.clear()
        assert len(cache) == 0
        assert cache.get("a") is None

    def test_make_key_deterministic(self):
        k1 = AsyncLRUCache.make_key("gpt-4", "hello", 0.7, 100)
        k2 = AsyncLRUCache.make_key("gpt-4", "hello", 0.7, 100)
        assert k1 == k2

    def test_make_key_different_params(self):
        k1 = AsyncLRUCache.make_key("gpt-4", "hello", 0.7, 100)
        k2 = AsyncLRUCache.make_key("gpt-4", "hello", 0.5, 100)
        assert k1 != k2


# ------------------------------------------------------------------ #
# retry_async
# ------------------------------------------------------------------ #


class TestRetryAsync:
    async def test_no_retry_succeeds(self):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await retry_async(fn, max_retries=0)
        assert result == "ok"
        assert call_count == 1

    async def test_retries_on_failure(self):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient error")
            return "ok"

        result = await retry_async(fn, max_retries=3, delay=0.01)
        assert result == "ok"
        assert call_count == 3

    async def test_raises_after_exhausting_retries(self):
        async def fn():
            raise RuntimeError("always fails")

        with pytest.raises(RuntimeError, match="always fails"):
            await retry_async(fn, max_retries=2, delay=0.01)

    async def test_no_retry_on_auth_error(self):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Invalid api_key provided")

        with pytest.raises(RuntimeError, match="api_key"):
            await retry_async(fn, max_retries=3, delay=0.01)
        assert call_count == 1  # Should not have retried

    async def test_no_retry_on_authentication_error(self):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Authentication failed")

        with pytest.raises(RuntimeError, match="Authentication"):
            await retry_async(fn, max_retries=3, delay=0.01)
        assert call_count == 1


# ------------------------------------------------------------------ #
# BaseLLM caching integration
# ------------------------------------------------------------------ #


class _DummyLLM(BaseLLM):
    """Minimal LLM for testing cache/retry integration."""

    def __init__(self, config: LLMConfig, responses: list[str] | None = None):
        super().__init__(config)
        self._responses = list(responses or ["default response"])
        self._call_count = 0

    async def stream(self, prompt: str, **kw):
        self._call_count += 1
        response = self._responses[min(self._call_count - 1, len(self._responses) - 1)]
        for word in response.split(" "):
            yield word + " "


class TestBaseLLMCaching:
    async def test_cache_disabled_by_default(self):
        config = LLMConfig(model="m", api_key="k", provider="test")
        llm = _DummyLLM(config)
        assert llm._cache is None

    async def test_cache_hit(self):
        config = LLMConfig(model="m", api_key="k", provider="test", cache=True)
        llm = _DummyLLM(config, ["first", "second"])

        r1 = await llm.generate("hello")
        r2 = await llm.generate("hello")  # should be cached
        assert r1 == r2
        assert llm._call_count == 1  # only called once

    async def test_cache_miss_different_prompt(self):
        config = LLMConfig(model="m", api_key="k", provider="test", cache=True)
        llm = _DummyLLM(config, ["first", "second"])

        await llm.generate("hello")
        await llm.generate("goodbye")
        assert llm._call_count == 2

    async def test_generate_with_messages_caching(self):
        config = LLMConfig(model="m", api_key="k", provider="test", cache=True)
        llm = _DummyLLM(config, ["answer"])

        msgs = [{"role": "user", "content": "hi"}]
        r1 = await llm.generate_with_messages(msgs)
        r2 = await llm.generate_with_messages(msgs)
        assert r1 == r2
        assert llm._call_count == 1


# ------------------------------------------------------------------ #
# BaseLLM retry integration
# ------------------------------------------------------------------ #


class _FailingLLM(BaseLLM):
    """LLM that fails N times then succeeds."""

    def __init__(self, config: LLMConfig, fail_count: int = 2):
        super().__init__(config)
        self._fail_count = fail_count
        self._call_count = 0

    async def stream(self, prompt: str, **kw):
        self._call_count += 1
        if self._call_count <= self._fail_count:
            raise RuntimeError("transient error")
        yield "success"


class TestBaseLLMRetry:
    async def test_retry_succeeds_after_failures(self):
        config = LLMConfig(
            model="m", api_key="k", provider="test", max_retries=3, retry_delay=0.01
        )
        llm = _FailingLLM(config, fail_count=2)
        result = await llm.generate("test")
        assert result == "success"
        assert llm._call_count == 3

    async def test_no_retry_by_default(self):
        config = LLMConfig(model="m", api_key="k", provider="test")
        llm = _FailingLLM(config, fail_count=1)
        with pytest.raises(RuntimeError, match="transient"):
            await llm.generate("test")
        assert llm._call_count == 1


# ------------------------------------------------------------------ #
# LLMConfig new defaults
# ------------------------------------------------------------------ #


def test_llmconfig_defaults():
    config = LLMConfig(model="m", api_key="k", provider="test")
    assert config.cache is False
    assert config.cache_maxsize == 128
    assert config.max_retries == 0
    assert config.retry_delay == 1.0
