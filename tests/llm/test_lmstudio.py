"""Tests for LMStudioLLM — mocked OpenAI-compatible client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synapsekit.llm.base import LLMConfig
from synapsekit.llm.lmstudio import _DEFAULT_BASE_URL, LMStudioLLM


def make_config(**kwargs):
    defaults = dict(model="llama-3-8b-instruct", api_key="lm-studio", provider="lmstudio")
    defaults.update(kwargs)
    return LLMConfig(**defaults)


def make_chunk(content=None, usage=None):
    chunk = MagicMock()
    if content is not None:
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = content
    else:
        chunk.choices = []
    chunk.usage = usage
    return chunk


# ------------------------------------------------------------------ #
# _get_client
# ------------------------------------------------------------------ #


def _make_mock_openai():
    """Return a mock openai module with a trackable AsyncOpenAI constructor."""
    mock_openai = MagicMock()
    mock_openai.AsyncOpenAI = MagicMock(return_value=MagicMock())
    return mock_openai


class TestGetClient:
    def test_uses_default_base_url(self):
        mock_openai = _make_mock_openai()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            llm = LMStudioLLM(make_config())
            llm._client = None
            llm._get_client()
            mock_openai.AsyncOpenAI.assert_called_once_with(
                base_url=_DEFAULT_BASE_URL, api_key="lm-studio"
            )

    def test_uses_custom_base_url(self):
        mock_openai = _make_mock_openai()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            llm = LMStudioLLM(make_config(), base_url="http://192.168.1.10:1234/v1")
            llm._client = None
            llm._get_client()
            mock_openai.AsyncOpenAI.assert_called_once_with(
                base_url="http://192.168.1.10:1234/v1", api_key="lm-studio"
            )

    def test_client_is_cached(self):
        mock_openai = _make_mock_openai()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            llm = LMStudioLLM(make_config())
            llm._client = None
            llm._get_client()
            llm._get_client()
            assert mock_openai.AsyncOpenAI.call_count == 1

    def test_missing_openai_raises_import_error(self):
        llm = LMStudioLLM(make_config())
        llm._client = None
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError, match="openai"):
                llm._get_client()


# ------------------------------------------------------------------ #
# stream
# ------------------------------------------------------------------ #


class TestStream:
    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        chunks = [make_chunk("Hello"), make_chunk(" world"), make_chunk(None)]

        async def async_chunks():
            for c in chunks:
                yield c

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=async_chunks())

        llm = LMStudioLLM(make_config())
        llm._client = mock_client

        tokens = []
        async for t in llm.stream("hi"):
            tokens.append(t)

        assert tokens == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_stream_skips_empty_content(self):
        chunks = [make_chunk(""), make_chunk(None), make_chunk("ok")]

        async def async_chunks():
            for c in chunks:
                yield c

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=async_chunks())

        llm = LMStudioLLM(make_config())
        llm._client = mock_client

        tokens = []
        async for t in llm.stream("hi"):
            tokens.append(t)

        assert tokens == ["ok"]

    @pytest.mark.asyncio
    async def test_stream_accumulates_token_usage(self):
        usage = MagicMock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 5
        chunks = [make_chunk(usage=usage), make_chunk("hi")]

        async def async_chunks():
            for c in chunks:
                yield c

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=async_chunks())

        llm = LMStudioLLM(make_config())
        llm._client = mock_client

        async for _ in llm.stream("hi"):
            pass

        assert llm.tokens_used["input"] == 10
        assert llm.tokens_used["output"] == 5

    @pytest.mark.asyncio
    async def test_stream_wraps_prompt_in_messages(self):
        async def async_chunks():
            yield make_chunk("ok")

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=async_chunks())

        llm = LMStudioLLM(make_config())
        llm._client = mock_client

        async for _ in llm.stream("test prompt"):
            pass

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1] == {"role": "user", "content": "test prompt"}


# ------------------------------------------------------------------ #
# stream_with_messages
# ------------------------------------------------------------------ #


class TestStreamWithMessages:
    @pytest.mark.asyncio
    async def test_stream_with_messages_yields_tokens(self):
        chunks = [make_chunk("A"), make_chunk("B")]

        async def async_chunks():
            for c in chunks:
                yield c

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=async_chunks())

        llm = LMStudioLLM(make_config())
        llm._client = mock_client

        tokens = []
        async for t in llm.stream_with_messages([{"role": "user", "content": "hi"}]):
            tokens.append(t)

        assert tokens == ["A", "B"]


# ------------------------------------------------------------------ #
# _call_with_tools_impl
# ------------------------------------------------------------------ #


class TestCallWithTools:
    @pytest.mark.asyncio
    async def test_returns_tool_calls(self):
        import json

        tool_call = MagicMock()
        tool_call.id = "call_abc"
        tool_call.function.name = "get_weather"
        tool_call.function.arguments = json.dumps({"location": "London"})

        msg = MagicMock()
        msg.tool_calls = [tool_call]
        msg.content = None

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = msg
        response.usage.prompt_tokens = 20
        response.usage.completion_tokens = 10

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=response)

        llm = LMStudioLLM(make_config())
        llm._client = mock_client

        result = await llm._call_with_tools_impl(
            messages=[{"role": "user", "content": "weather?"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )

        assert result["content"] is None
        assert result["tool_calls"][0]["name"] == "get_weather"
        assert result["tool_calls"][0]["arguments"] == {"location": "London"}
        assert result["tool_calls"][0]["id"] == "call_abc"

    @pytest.mark.asyncio
    async def test_returns_content_when_no_tool_call(self):
        msg = MagicMock()
        msg.tool_calls = None
        msg.content = "Just a plain response."

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = msg
        response.usage.prompt_tokens = 5
        response.usage.completion_tokens = 8

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=response)

        llm = LMStudioLLM(make_config())
        llm._client = mock_client

        result = await llm._call_with_tools_impl(
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
        )

        assert result["content"] == "Just a plain response."
        assert result["tool_calls"] is None


# ------------------------------------------------------------------ #
# Coroutine function checks
# ------------------------------------------------------------------ #


def test_stream_is_async_generator():
    import inspect

    llm = LMStudioLLM(make_config())
    assert inspect.isasyncgenfunction(llm.stream)


def test_stream_with_messages_is_async_generator():
    import inspect

    llm = LMStudioLLM(make_config())
    assert inspect.isasyncgenfunction(llm.stream_with_messages)
