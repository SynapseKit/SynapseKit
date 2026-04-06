"""Tests for WriterLLM provider — mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from synapsekit.llm.base import LLMConfig
from synapsekit.llm.writer import WriterLLM


def make_config(model: str = "palmyra-x-004") -> LLMConfig:
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
        c1 = MagicMock()
        c1.choices = [MagicMock()]
        c1.choices[0].delta.content = "Hello"
        c2 = MagicMock()
        c2.choices = [MagicMock()]
        c2.choices[0].delta.content = " Writer"

        async def stream_gen(**kw):
            for c in [c1, c2]:
                yield c

        async def mock_create(**kw):
            return stream_gen(**kw)

        mock_completions = MagicMock()
        mock_completions.create = mock_create
        mock_chat = MagicMock()
        mock_chat.completions = mock_completions
        mock_client = MagicMock()
        mock_client.chat = mock_chat

        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"openai": mock_openai}):
            llm = WriterLLM(make_config())
            got = []
            async for t in llm.stream("hi"):
                got.append(t)
            assert got == ["Hello", " Writer"]

    @pytest.mark.asyncio
    async def test_call_with_tools(self):
        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "lookup"
        tool_call.function.arguments = '{"q":"abc"}'

        msg = MagicMock()
        msg.content = None
        msg.tool_calls = [tool_call]

        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message = msg
        resp.usage = MagicMock()
        resp.usage.prompt_tokens = 4
        resp.usage.completion_tokens = 2

        async def mock_create(**kw):
            return resp

        mock_completions = MagicMock()
        mock_completions.create = mock_create
        mock_chat = MagicMock()
        mock_chat.completions = mock_completions
        mock_client = MagicMock()
        mock_client.chat = mock_chat

        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"openai": mock_openai}):
            llm = WriterLLM(make_config("palmyra-x-003-instruct"))
            out = await llm._call_with_tools_impl(
                messages=[{"role": "user", "content": "find"}],
                tools=[{"type": "function", "function": {"name": "lookup"}}],
            )
            assert out["content"] is None
            assert out["tool_calls"][0]["arguments"] == {"q": "abc"}
            assert llm._input_tokens == 4
            assert llm._output_tokens == 2

    def test_default_base_url(self):
        llm = WriterLLM(make_config())
        assert llm._base_url == "https://api.writer.com/v1"

    def test_make_llm_detects_palmyra_prefix(self):
        from synapsekit.rag.facade import _make_llm

        mock_openai = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            llm = _make_llm("palmyra-fin", "key", None, "sys", 0.2, 100)
            assert isinstance(llm, WriterLLM)
