"""Tests for VLLMLLM provider (mocked OpenAI-compatible client)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synapsekit.llm.base import LLMConfig


def _config(model: str = "meta-llama/Llama-3.1-8B-Instruct") -> LLMConfig:
    return LLMConfig(
        model=model,
        api_key="test-key",
        provider="vllm",
        system_prompt="You are helpful.",
        temperature=0.2,
        max_tokens=100,
    )


def _mock_stream_response(texts: list[str], usage: tuple[int, int] | None = None):
    chunks = []
    if usage is not None:
        usage_chunk = MagicMock()
        usage_chunk.choices = []
        usage_chunk.usage = MagicMock()
        usage_chunk.usage.prompt_tokens = usage[0]
        usage_chunk.usage.completion_tokens = usage[1]
        chunks.append(usage_chunk)

    for text in texts:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = text
        chunk.usage = None
        chunks.append(chunk)

    async def async_iter():
        for c in chunks:
            yield c

    mock_response = MagicMock()
    mock_response.__aiter__ = lambda self: async_iter()
    return mock_response


class TestVLLMLLM:
    def test_import_from_top_level(self):
        from synapsekit import VLLMLLM

        assert VLLMLLM is not None

    def test_construction_uses_default_base_url(self):
        from synapsekit.llm.vllm import _DEFAULT_BASE_URL, VLLMLLM

        llm = VLLMLLM(_config())
        assert llm._base_url == _DEFAULT_BASE_URL

    def test_custom_base_url(self):
        from synapsekit.llm.vllm import VLLMLLM

        llm = VLLMLLM(_config(), base_url="http://10.0.0.5:8000/v1")
        assert llm._base_url == "http://10.0.0.5:8000/v1"

    def test_import_error_without_openai(self):
        from synapsekit.llm.vllm import VLLMLLM

        llm = VLLMLLM(_config())
        with patch.dict("sys.modules", {"openai": None}):
            llm._client = None
            with pytest.raises(ImportError, match="openai"):
                llm._get_client()

    def test_get_client_uses_base_url_and_api_key(self):
        from synapsekit.llm.vllm import VLLMLLM

        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI = MagicMock(return_value=MagicMock())

        with patch.dict("sys.modules", {"openai": mock_openai}):
            llm = VLLMLLM(_config(), base_url="http://localhost:9000/v1")
            llm._client = None
            llm._get_client()
            mock_openai.AsyncOpenAI.assert_called_once_with(
                api_key="test-key",
                base_url="http://localhost:9000/v1",
            )

    @pytest.mark.asyncio
    async def test_stream_yields_tokens_and_tracks_usage(self):
        from synapsekit.llm.vllm import VLLMLLM

        llm = VLLMLLM(_config())

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_stream_response(["Hello", " vLLM"], usage=(11, 7))
        )
        llm._client = mock_client

        tokens = []
        async for t in llm.stream("hi"):
            tokens.append(t)

        assert tokens == ["Hello", " vLLM"]
        assert llm.tokens_used == {"input": 11, "output": 7}

    @pytest.mark.asyncio
    async def test_call_with_tools_returns_parsed_args(self):
        from synapsekit.llm.vllm import VLLMLLM

        llm = VLLMLLM(_config())

        tc = MagicMock()
        tc.id = "call_1"
        tc.function.name = "calculator"
        tc.function.arguments = json.dumps({"expr": "2+2"})

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = MagicMock(content=None, tool_calls=[tc])
        response.usage.prompt_tokens = 9
        response.usage.completion_tokens = 4

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
        assert llm.tokens_used == {"input": 9, "output": 4}


def test_facade_explicit_vllm():
    from synapsekit.rag.facade import _make_llm

    mock_openai = MagicMock()
    with patch.dict("sys.modules", {"openai": mock_openai}):
        from synapsekit.llm.vllm import VLLMLLM

        llm = _make_llm("meta-llama/Llama-3.1-8B-Instruct", "key", "vllm", "sys", 0.2, 100)
        assert isinstance(llm, VLLMLLM)
