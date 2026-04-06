"""Tests for WriterLLM provider — mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from synapsekit.llm.base import LLMConfig
from synapsekit.llm.writer import WriterLLM


def make_config(model="palmyra-x-004"):
    return LLMConfig(
        model=model,
        api_key="test-writer-key",
        provider="writer",
        system_prompt="You are helpful.",
        temperature=0.2,
        max_tokens=100,
    )


class TestWriterLLM:
    def test_import_error_without_openai(self):
        with patch.dict("sys.modules", {"openai": None}):
            llm = WriterLLM(make_config())
            llm._client = None
            with pytest.raises(ImportError, match="openai package required"):
                llm._get_client()

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta.content = "Hello"

        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta.content = " from Writer"

        async def mock_create_stream(**kw):
            for chunk in [mock_chunk1, mock_chunk2]:
                yield chunk

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
            llm = WriterLLM(make_config())
            tokens = []
            async for t in llm.stream("hello"):
                tokens.append(t)
            assert tokens == ["Hello", " from Writer"]

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
            llm = WriterLLM(make_config())
            tokens = []
            messages = [{"role": "user", "content": "hi"}]
            async for t in llm.stream_with_messages(messages):
                tokens.append(t)
            assert tokens == ["ok"]

    @pytest.mark.asyncio
    async def test_call_with_tools_impl_no_tool_calls(self):
        mock_message = MagicMock()
        mock_message.content = "Here you go!"
        mock_message.tool_calls = None

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = mock_message
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 12
        mock_response.usage.completion_tokens = 8

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
            llm = WriterLLM(make_config())
            result = await llm._call_with_tools_impl(
                messages=[{"role": "user", "content": "hello"}],
                tools=[{"type": "function", "function": {"name": "get_info"}}],
            )
            assert result["content"] == "Here you go!"
            assert result["tool_calls"] is None
            assert llm._input_tokens == 12
            assert llm._output_tokens == 8

    @pytest.mark.asyncio
    async def test_call_with_tools_impl_with_tool_calls(self):
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_xyz"
        mock_tool_call.function.name = "search"
        mock_tool_call.function.arguments = '{"query": "AI news"}'

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = mock_message
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 6

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
            llm = WriterLLM(make_config("palmyra-x-003-instruct"))
            result = await llm._call_with_tools_impl(
                messages=[{"role": "user", "content": "find news"}],
                tools=[{"type": "function", "function": {"name": "search"}}],
            )
            assert result["content"] is None
            assert len(result["tool_calls"]) == 1
            assert result["tool_calls"][0]["id"] == "call_xyz"
            assert result["tool_calls"][0]["name"] == "search"
            assert result["tool_calls"][0]["arguments"] == {"query": "AI news"}

    def test_custom_base_url(self):
        llm = WriterLLM(make_config(), base_url="https://custom.writer.com/v1")
        assert llm._base_url == "https://custom.writer.com/v1"

    def test_default_base_url(self):
        llm = WriterLLM(make_config())
        assert llm._base_url == "https://api.writer.com/v1"
