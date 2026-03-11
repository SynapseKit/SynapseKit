"""Tests for the RAG facade — 3-line API."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from synapsekit import RAG
from synapsekit.llm.base import LLMConfig
from synapsekit.observability.tracer import TokenTracer
from synapsekit.memory.conversation import ConversationMemory


def _patch_rag(rag: RAG, tokens=("Answer", " here")):
    """Replace the pipeline's LLM and retriever with mocks."""
    # Patch retriever
    rag._pipeline.config.retriever.retrieve = AsyncMock(return_value=["context"])
    rag._pipeline.config.retriever._store.add = AsyncMock()

    # Patch LLM stream
    async def mock_stream(messages, **kw):
        for t in tokens:
            yield t

    rag._pipeline.config.llm.stream_with_messages = mock_stream
    rag._pipeline.config.llm._input_tokens = 5
    rag._pipeline.config.llm._output_tokens = 3

    # Patch splitter
    mock_splitter = MagicMock()
    mock_splitter.split = MagicMock(return_value=["chunk"])
    rag._pipeline._splitter = mock_splitter


class TestRAGFacade:
    def test_init_openai_auto_detect(self):
        with patch("synapsekit.llm.openai.OpenAILLM.__init__", return_value=None):
            rag = RAG(model="gpt-4o-mini", api_key="sk-test")
            assert rag._pipeline.config.llm is not None

    def test_init_anthropic_auto_detect(self):
        with patch("synapsekit.llm.anthropic.AnthropicLLM.__init__", return_value=None):
            rag = RAG(model="claude-haiku-4-5-20251001", api_key="sk-test")
            assert rag._pipeline.config.llm is not None

    def test_init_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            RAG(model="gpt-4o-mini", api_key="sk-test", provider="unknownxyz")

    def test_add_sync(self):
        rag = RAG(model="gpt-4o-mini", api_key="sk-test")
        _patch_rag(rag)
        rag.add("Some document text.")
        rag._pipeline._splitter.split.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_async(self):
        rag = RAG(model="gpt-4o-mini", api_key="sk-test")
        _patch_rag(rag)
        await rag.add_async("Some document text.")
        rag._pipeline._splitter.split.assert_called_once()

    @pytest.mark.asyncio
    async def test_ask_returns_string(self):
        rag = RAG(model="gpt-4o-mini", api_key="sk-test")
        _patch_rag(rag)
        answer = await rag.ask("What is the topic?")
        assert answer == "Answer here"

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        rag = RAG(model="gpt-4o-mini", api_key="sk-test")
        _patch_rag(rag, tokens=["A", "B", "C"])
        tokens = []
        async for t in rag.stream("question?"):
            tokens.append(t)
        assert tokens == ["A", "B", "C"]

    def test_ask_sync(self):
        rag = RAG(model="gpt-4o-mini", api_key="sk-test")
        _patch_rag(rag)
        answer = rag.ask_sync("question?")
        assert answer == "Answer here"

    def test_tracer_property(self):
        rag = RAG(model="gpt-4o-mini", api_key="sk-test")
        assert isinstance(rag.tracer, TokenTracer)

    def test_memory_property(self):
        rag = RAG(model="gpt-4o-mini", api_key="sk-test")
        assert isinstance(rag.memory, ConversationMemory)

    def test_save_raises_when_empty(self, tmp_path):
        rag = RAG(model="gpt-4o-mini", api_key="sk-test")
        with pytest.raises(ValueError, match="empty"):
            rag.save(str(tmp_path / "store.npz"))

    @pytest.mark.asyncio
    async def test_trace_disabled(self):
        rag = RAG(model="gpt-4o-mini", api_key="sk-test", trace=False)
        _patch_rag(rag)
        await rag.ask("test")
        assert rag.tracer.summary()["calls"] == 0
