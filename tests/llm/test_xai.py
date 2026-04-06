"""Tests for XaiLLM provider — mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from synapsekit.llm.base import LLMConfig
from synapsekit.llm.xai import XaiLLM


def make_config(model="grok-beta"):
    return LLMConfig(
        model=model,
        api_key="test-xai-key",
        provider="xai",
        system_prompt="You are helpful.",
        temperature=0.2,
        max_tokens=100,
    )


class TestXaiLLM:
    def test_import_error_without_openai(self):
        with patch.dict("sys.modules", {"openai": None}):
            llm = XaiLLM(make_config())
            llm._client = None  # force re-init
            with pytest.raises(ImportError, match="openai package required"):
                llm._get_client()

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta.content = "Hello"

        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta.content = " from Grok"

        async def mock_create_stream(**kw):
            for chunk in [mock_chunk1, mock_chunk2]:
                yield chunk

        # Create a mock that makes create() awaitable and return the generator
        async def mock_create(**kw):
            return mock_create_stream(**kw)

        mock_completions = MagicMock()
        mock_completions.create = mock_create
        mock_chat = MagicMock()
        mock_chat.completions = mock_completions
        mock_client = MagicMock()
        mock_client.chat = mock_chat

        mock_async_openai = MagicMock(return_value=mock_client)
        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI = mock_async_openai

        with patch.dict("sys.modules", {"openai": mock_openai}):
            llm = XaiLLM(make_config("grok-beta"))
            tokens = []
            async for t in llm.stream("hello"):
                tokens.append(t)
            assert tokens == ["Hello", " from Grok"]

    @pytest.mark.asyncio
    async def test_stream_with_messages(self):
        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = "ok"

        async def mock_create_stream(**kw):
            yield mock_chunk

        async def mock_create(**kw):
            return mock_create_stream(**kw)

        mock_completions = MagicMock()
        mock_completions.create = mock_create
        mock_chat = MagicMock()
        mock_chat.completions = mock_completions
        mock_client = MagicMock()
        mock_client.chat = mock_chat

        mock_async_openai = MagicMock(return_value=mock_client)
        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI = mock_async_openai

        with patch.dict("sys.modules", {"openai": mock_openai}):
            llm = XaiLLM(make_config())
            tokens = []
            messages = [{"role": "user", "content": "hi"}]
            async for t in llm.stream_with_messages(messages):
                tokens.append(t)
            assert tokens == ["ok"]

    @pytest.mark.asyncio
    async def test_call_with_tools_impl_no_tool_calls(self):
        mock_message = MagicMock()
        mock_message.content = "Sure, here you go!"
        mock_message.tool_calls = None

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = mock_message
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20

        async def mock_create(**kw):
            return mock_response

        mock_completions = MagicMock()
        mock_completions.create = mock_create
        mock_chat = MagicMock()
        mock_chat.completions = mock_completions
        mock_client = MagicMock()
        mock_client.chat = mock_chat

        mock_async_openai = MagicMock(return_value=mock_client)
        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI = mock_async_openai

        with patch.dict("sys.modules", {"openai": mock_openai}):
            llm = XaiLLM(make_config("grok-2"))
            result = await llm._call_with_tools_impl(
                messages=[{"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {"name": "get_weather"}}],
            )
            assert result["content"] == "Sure, here you go!"
            assert result["tool_calls"] is None
            assert llm._input_tokens == 10
            assert llm._output_tokens == 20

    @pytest.mark.asyncio
    async def test_call_with_tools_impl_with_tool_calls(self):
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = '{"location": "NYC"}'

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = mock_message
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 15
        mock_response.usage.completion_tokens = 5

        async def mock_create(**kw):
            return mock_response

        mock_completions = MagicMock()
        mock_completions.create = mock_create
        mock_chat = MagicMock()
        mock_chat.completions = mock_completions
        mock_client = MagicMock()
        mock_client.chat = mock_chat

        mock_async_openai = MagicMock(return_value=mock_client)
        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI = mock_async_openai

        with patch.dict("sys.modules", {"openai": mock_openai}):
            llm = XaiLLM(make_config("grok-2-mini"))
            result = await llm._call_with_tools_impl(
                messages=[{"role": "user", "content": "weather in NYC?"}],
                tools=[{"type": "function", "function": {"name": "get_weather"}}],
            )
            assert result["content"] is None
            assert len(result["tool_calls"]) == 1
            assert result["tool_calls"][0]["id"] == "call_123"
            assert result["tool_calls"][0]["name"] == "get_weather"
            assert result["tool_calls"][0]["arguments"] == {"location": "NYC"}
            assert llm._input_tokens == 15
            assert llm._output_tokens == 5

    def test_custom_base_url(self):
        config = make_config()
        llm = XaiLLM(config, base_url="https://custom.x.ai/v1")
        assert llm._base_url == "https://custom.x.ai/v1"

    def test_default_base_url(self):
        config = make_config()
        llm = XaiLLM(config)
        assert llm._base_url == "https://api.x.ai/v1"
