"""Tests for LLM layer — OpenAI and Anthropic with mocked clients."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from synapsekit.llm.base import LLMConfig, _messages_to_prompt
from synapsekit.llm.openai import OpenAILLM
from synapsekit.llm.anthropic import AnthropicLLM


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
