"""Tests for NovitaLLM provider — mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from synapsekit.llm.base import LLMConfig
from synapsekit.llm.novita import NovitaLLM


def make_config(model="meta-llama/llama-3.1-8b-instruct"):
    return LLMConfig(
        model=model,
        api_key="test-novita-key",
        provider="novita",
        system_prompt="You are helpful.",
        temperature=0.2,
        max_tokens=100,
    )


class TestNovitaLLM:
    def test_import_error_without_openai(self):
        with patch.dict("sys.modules", {"openai": None}):
            llm = NovitaLLM(make_config())
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
        mock_chunk2.choices[0].delta.content = " from Novita"

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
            llm = NovitaLLM(make_config())
            tokens = []
            async for t in llm.stream("hello"):
                tokens.append(t)
            assert tokens == ["Hello", " from Novita"]

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
            llm = NovitaLLM(make_config())
            tokens = []
            messages = [{"role": "user", "content": "hi"}]
            async for t in llm.stream_with_messages(messages):
                tokens.append(t)
            assert tokens == ["ok"]

    @pytest.mark.asyncio
    async def test_call_with_tools_impl_no_tool_calls(self):
        mock_message = MagicMock()
        mock_message.content = "Sure!"
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
            llm = NovitaLLM(make_config())
            result = await llm._call_with_tools_impl(
                messages=[{"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {"name": "calc"}}],
            )
            assert result["content"] == "Sure!"
            assert result["tool_calls"] is None

    @pytest.mark.asyncio
    async def test_call_with_tools_impl_with_tool_calls(self):
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_abc"
        mock_tool_call.function.name = "calculate"
        mock_tool_call.function.arguments = '{"x": 5}'

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = mock_message
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 8
        mock_response.usage.completion_tokens = 4

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
            llm = NovitaLLM(make_config())
            result = await llm._call_with_tools_impl(
                messages=[{"role": "user", "content": "calc"}],
                tools=[{"type": "function", "function": {"name": "calculate"}}],
            )
            assert result["content"] is None
            assert len(result["tool_calls"]) == 1
            assert result["tool_calls"][0]["id"] == "call_abc"
            assert result["tool_calls"][0]["name"] == "calculate"
            assert result["tool_calls"][0]["arguments"] == {"x": 5}

    def test_custom_base_url(self):
        llm = NovitaLLM(make_config(), base_url="https://custom.novita.ai/v3/openai")
        assert llm._base_url == "https://custom.novita.ai/v3/openai"

    def test_default_base_url(self):
        llm = NovitaLLM(make_config())
        assert llm._base_url == "https://api.novita.ai/v3/openai"
