"""Tests for RAGPipeline — end-to-end with mocks."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from synapsekit.rag.pipeline import RAGConfig, RAGPipeline
from synapsekit.memory.conversation import ConversationMemory
from synapsekit.observability.tracer import TokenTracer


def make_mock_llm(tokens=("Hello", " world")):
    llm = MagicMock()
    llm.tokens_used = {"input": 10, "output": 5}

    async def stream_with_messages(messages, **kw):
        for t in tokens:
            yield t

    llm.stream_with_messages = stream_with_messages
    return llm


def make_mock_retriever(chunks=None):
    retriever = MagicMock()
    retriever.retrieve = AsyncMock(return_value=chunks or ["Context chunk 1.", "Context chunk 2."])
    retriever.add = AsyncMock()
    return retriever


@pytest.fixture
def pipeline():
    llm = make_mock_llm()
    retriever = make_mock_retriever()
    memory = ConversationMemory()
    tracer = TokenTracer(model="gpt-4o-mini")

    config = RAGConfig(
        llm=llm,
        retriever=retriever,
        memory=memory,
        tracer=tracer,
    )
    return RAGPipeline(config)


class TestRAGPipeline:
    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self, pipeline):
        tokens = []
        async for token in pipeline.stream("What is this?"):
            tokens.append(token)
        assert tokens == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_ask_returns_string(self, pipeline):
        answer = await pipeline.ask("What is this?")
        assert answer == "Hello world"

    @pytest.mark.asyncio
    async def test_memory_updated_after_stream(self, pipeline):
        await pipeline.ask("My question?")
        messages = pipeline.config.memory.get_messages()
        assert any(m["content"] == "My question?" for m in messages)
        assert any(m["content"] == "Hello world" for m in messages)

    @pytest.mark.asyncio
    async def test_tracer_records_after_stream(self, pipeline):
        await pipeline.ask("test?")
        s = pipeline.config.tracer.summary()
        assert s["calls"] == 1

    @pytest.mark.asyncio
    async def test_add_calls_splitter_and_store(self, pipeline):
        mock_splitter = MagicMock()
        mock_splitter.split = MagicMock(return_value=["chunk1", "chunk2"])
        pipeline._splitter = mock_splitter

        await pipeline.add("Some long text to chunk.")
        mock_splitter.split.assert_called_once()
        pipeline.config.retriever.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_retrieval_uses_no_context_message(self, pipeline):
        pipeline.config.retriever.retrieve = AsyncMock(return_value=[])
        tokens = []
        async for token in pipeline.stream("test?"):
            tokens.append(token)
        assert len(tokens) > 0  # LLM still responds

    @pytest.mark.asyncio
    async def test_add_chunks_text(self):
        llm = make_mock_llm()
        retriever = make_mock_retriever()
        pipeline = RAGPipeline(RAGConfig(llm=llm, retriever=retriever, memory=ConversationMemory()))
        # add should not raise — TextSplitter is pure Python
        await pipeline.add("Hello world. This is a test document.")
