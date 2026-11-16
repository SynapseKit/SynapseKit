"""Tests for LLM layer — OpenAI and Anthropic with mocked clients."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synapsekit.llm.anthropic import AnthropicLLM
from synapsekit.llm.base import LLMConfig, _messages_to_prompt
from synapsekit.llm.openai import OpenAILLM

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def make_openai_config(model="gpt-4o-mini"):
    return LLMConfig(model=model, api_key="test-key", provider="openai")


def make_anthropic_config(model="claude-haiku-4-5-20251001"):
    return LLMConfig(model=model, api_key="test-key", provider="anthropic")


# ------------------------------------------------------------------ #
# BaseLLM
# ------------------------------------------------------------------ #


class TestBaseLLM:
    def test_messages_to_prompt_formats_correctly(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = _messages_to_prompt(messages)
        assert "User: Hello" in result
        assert "Assistant: Hi there" in result

    def test_tokens_used_initial_zero(self):
        llm = OpenAILLM(make_openai_config())
        assert llm.tokens_used == {"input": 0, "output": 0}


# ------------------------------------------------------------------ #
# OpenAILLM
# ------------------------------------------------------------------ #


class TestOpenAILLM:
    @pytest.fixture
    def llm(self):
        return OpenAILLM(make_openai_config())

    def _make_chunk(self, content=None, usage=None):
        chunk = MagicMock()
        if content is not None:
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = content
        else:
            chunk.choices = []
        chunk.usage = usage
        return chunk

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self, llm):
        raw_chunks = [
            self._make_chunk("Hello"),
            self._make_chunk(" world"),
            self._make_chunk(None),
        ]

        async def async_chunks():
            for c in raw_chunks:
                yield c

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=async_chunks())
        llm._client = mock_client

        tokens = []
        async for token in llm.stream("test prompt"):
            tokens.append(token)

        assert tokens == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_generate_collects_tokens(self, llm):
        raw_chunks = [self._make_chunk("Hi"), self._make_chunk("!")]

        async def async_chunks():
            for c in raw_chunks:
                yield c

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=async_chunks())
        llm._client = mock_client

        result = await llm.generate("test")
        assert result == "Hi!"

    def test_missing_openai_raises(self):
        llm = OpenAILLM(make_openai_config())
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError, match="openai"):
                llm._get_client()


# ------------------------------------------------------------------ #
# AnthropicLLM
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
@pytest.mark.parametrize("max_retries", [0, 2])
@pytest.mark.parametrize("operation", ["generate", "messages", "tools", "stream"])
async def test_openai_retry_count_is_not_multiplied_by_sdk(monkeypatch, max_retries, operation):
    httpx = pytest.importorskip("httpx")
    openai = pytest.importorskip("openai")

    attempts = []

    def respond(request):
        attempts.append(request)
        return httpx.Response(
            429, headers={"retry-after": "0"}, json={"error": {"message": "busy"}}
        )

    sdk_client = openai.AsyncOpenAI
    monkeypatch.setattr(
        openai,
        "AsyncOpenAI",
        lambda **kw: sdk_client(
            **kw, http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
        ),
    )
    llm = OpenAILLM(
        LLMConfig(model="m", api_key="k", provider="openai", max_retries=max_retries, retry_delay=0)
    )
    try:
        with pytest.raises(openai.RateLimitError):
            if operation == "generate":
                await llm.generate("q")
            elif operation == "messages":
                await llm.generate_with_messages([{"role": "user", "content": "q"}])
            elif operation == "tools":
                await llm.call_with_tools([{"role": "user", "content": "q"}], [])
            else:
                _ = [chunk async for chunk in llm.stream("q")]
        assert len(attempts) == (1 if operation == "stream" else max_retries + 1)
        assert llm._client.max_retries == 0
    finally:
        await llm.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("max_retries", [0, 2])
@pytest.mark.parametrize("failure", ["connect", "timeout"])
async def test_openai_transport_failures_use_base_retry_budget(monkeypatch, max_retries, failure):
    httpx = pytest.importorskip("httpx")
    openai = pytest.importorskip("openai")

    attempts = []

    def respond(request):
        attempts.append(request)
        if failure == "connect":
            raise httpx.ConnectError("offline", request=request)
        raise httpx.ReadTimeout("timed out", request=request)

    sdk_client = openai.AsyncOpenAI

    def make_client(**kwargs):
        return sdk_client(
            **kwargs,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        )

    monkeypatch.setattr(openai, "AsyncOpenAI", make_client)
    llm = OpenAILLM(
        LLMConfig(model="m", api_key="k", provider="openai", max_retries=max_retries, retry_delay=0)
    )
    try:
        with pytest.raises(openai.APIConnectionError):
            await llm.generate("q")
        assert len(attempts) == max_retries + 1
        assert llm._client.max_retries == 0
    finally:
        await llm.aclose()


class TestAnthropicLLM:
    @pytest.fixture
    def llm(self):
        return AnthropicLLM(make_anthropic_config())

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self, llm):
        tokens_to_yield = ["Hello", " there"]

        mock_message = MagicMock()
        mock_message.usage.input_tokens = 10
        mock_message.usage.output_tokens = 5

        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_ctx)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        async def text_stream_gen():
            for t in tokens_to_yield:
                yield t

        mock_stream_ctx.text_stream = text_stream_gen()
        mock_stream_ctx.get_final_message = AsyncMock(return_value=mock_message)

        mock_client = MagicMock()
        mock_client.messages.stream = MagicMock(return_value=mock_stream_ctx)
        llm._client = mock_client

        collected = []
        async for token in llm.stream("test prompt"):
            collected.append(token)

        assert collected == ["Hello", " there"]
        assert llm.tokens_used["input"] == 10
        assert llm.tokens_used["output"] == 5

    def test_missing_anthropic_raises(self):
        llm = AnthropicLLM(make_anthropic_config())
        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises(ImportError, match="anthropic"):
                llm._get_client()
