"""Tests for SambaNovaLLM provider (mocked OpenAI-compatible client)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synapsekit.llm.base import LLMConfig


def _config(model: str = "Meta-Llama-3.1-8B-Instruct") -> LLMConfig:
    return LLMConfig(
        model=model,
        api_key="test-key",
        provider="sambanova",
        system_prompt="You are helpful.",
        temperature=0.2,
        max_tokens=100,
    )


def _mock_stream_response(texts: list[str]):
    chunks = []
    for text in texts:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = text
        chunks.append(chunk)

    async def async_iter():
        for c in chunks:
            yield c

    mock_response = MagicMock()
    mock_response.__aiter__ = lambda self: async_iter()
    return mock_response


class TestSambaNovaLLM:
    def test_import_from_top_level(self):
        from synapsekit import SambaNovaLLM

        assert SambaNovaLLM is not None

    def test_construction_uses_default_base_url(self):
        from synapsekit.llm.sambanova import _SAMBANOVA_BASE_URL, SambaNovaLLM

        llm = SambaNovaLLM(_config())
        assert llm._base_url == _SAMBANOVA_BASE_URL

    def test_custom_base_url(self):
        from synapsekit.llm.sambanova import SambaNovaLLM

        llm = SambaNovaLLM(_config(), base_url="https://custom.sambanova.local/v1")
        assert llm._base_url == "https://custom.sambanova.local/v1"

    def test_import_error_without_openai(self):
        from synapsekit.llm.sambanova import SambaNovaLLM

        llm = SambaNovaLLM(_config())
        with patch.dict("sys.modules", {"openai": None}):
            llm._client = None
            with pytest.raises(ImportError, match="openai"):
                llm._get_client()

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        from synapsekit.llm.sambanova import SambaNovaLLM

        llm = SambaNovaLLM(_config())

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_stream_response(["Hello", " SambaNova"])
        )
        llm._client = mock_client

        tokens = []
        async for t in llm.stream("hi"):
            tokens.append(t)

        assert tokens == ["Hello", " SambaNova"]

    @pytest.mark.asyncio
    async def test_generate_collects_stream(self):
        from synapsekit.llm.sambanova import SambaNovaLLM

        llm = SambaNovaLLM(_config())

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_stream_response(["All", " good"])
        )
        llm._client = mock_client

        result = await llm.generate("hi")
        assert result == "All good"

    @pytest.mark.asyncio
    async def test_call_with_tools_returns_parsed_args(self):
        from synapsekit.llm.sambanova import SambaNovaLLM

        llm = SambaNovaLLM(_config())

        tc = MagicMock()
        tc.id = "call_1"
        tc.function.name = "calculator"
        tc.function.arguments = json.dumps({"expr": "2+2"})

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = MagicMock(content=None, tool_calls=[tc])
        response.usage.prompt_tokens = 10
        response.usage.completion_tokens = 5

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=response)
        llm._client = mock_client

        result = await llm._call_with_tools_impl(
            [{"role": "user", "content": "calc"}],
            [{"type": "function", "function": {"name": "calculator"}}],
        )

        assert result["content"] is None
        assert result["tool_calls"][0]["name"] == "calculator"
        assert result["tool_calls"][0]["arguments"] == {"expr": "2+2"}
        assert llm.tokens_used == {"input": 10, "output": 5}


def test_facade_explicit_sambanova():
    from synapsekit.rag.facade import _make_llm

    mock_openai = MagicMock()
    with patch.dict("sys.modules", {"openai": mock_openai}):
        from synapsekit.llm.sambanova import SambaNovaLLM

        llm = _make_llm("Meta-Llama-3.1-8B-Instruct", "key", "sambanova", "sys", 0.2, 100)
        assert isinstance(llm, SambaNovaLLM)
