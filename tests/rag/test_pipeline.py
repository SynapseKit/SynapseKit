"""Tests for RAGPipeline — end-to-end with mocks."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from synapsekit.memory.conversation import ConversationMemory
from synapsekit.observability.tracer import TokenTracer
from synapsekit.rag.pipeline import RAGConfig, RAGPipeline


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
    async def test_stream_commits_memory_on_consumer_disconnect(self, pipeline):
        """Consumer breaks after 1 token and explicitly closes the generator —
        simulates a streaming-HTTP client disconnect, which causes the ASGI
        server (starlette/anyio) to call aclose() on the response generator.
        Memory must still reflect the query and the partial answer the
        consumer saw. Fails before the fix."""
        seen = []
        gen = pipeline.stream("Partial question?")
        try:
            async for token in gen:
                seen.append(token)
                break  # simulate consumer stopping iteration
        finally:
            await gen.aclose()  # simulate ASGI-level disconnect cleanup

        assert seen == ["Hello"]
        messages = pipeline.config.memory.get_messages()
        contents = [m["content"] for m in messages]
        assert "Partial question?" in contents
        assert "Hello" in contents  # partial answer preserved

    @pytest.mark.asyncio
    async def test_stream_no_memory_commit_on_pre_stream_failure(self):
        """If the LLM fails before yielding any token, memory should not
        record a ghost turn with an empty assistant response."""
        llm = MagicMock()
        llm.tokens_used = {"input": 0, "output": 0}

        async def failing_stream(messages, **kw):
            raise RuntimeError("auth error")
            yield  # unreachable; marks this as an async generator

        llm.stream_with_messages = failing_stream

        retriever = make_mock_retriever()
        memory = ConversationMemory()
        pipeline = RAGPipeline(RAGConfig(llm=llm, retriever=retriever, memory=memory))

        with pytest.raises(RuntimeError, match="auth error"):
            async for _ in pipeline.stream("Never streams."):
                pass

        assert len(memory) == 0, "No memory should be recorded when no tokens were emitted"

    @pytest.mark.asyncio
    async def test_stream_commits_partial_answer_on_mid_stream_llm_failure(self):
        """If the LLM yields some tokens then raises mid-stream (e.g. transient
        network error), memory must capture the partial answer the consumer
        already saw, not silently drop it."""
        llm = MagicMock()
        llm.tokens_used = {"input": 10, "output": 5}

        async def partial_failure_stream(messages, **kw):
            yield "Partial "
            yield "answer"
            raise RuntimeError("connection reset")

        llm.stream_with_messages = partial_failure_stream

        retriever = make_mock_retriever()
        memory = ConversationMemory()
        pipeline = RAGPipeline(RAGConfig(llm=llm, retriever=retriever, memory=memory))

        seen = []
        with pytest.raises(RuntimeError, match="connection reset"):
            async for token in pipeline.stream("Mid-stream failure?"):
                seen.append(token)

        assert seen == ["Partial ", "answer"]
        contents = [m["content"] for m in memory.get_messages()]
        assert "Mid-stream failure?" in contents
        assert "Partial answer" in contents  # partial answer preserved despite LLM error

    @pytest.mark.asyncio
    async def test_add_chunks_text(self):
        llm = make_mock_llm()
        retriever = make_mock_retriever()
        pipeline = RAGPipeline(RAGConfig(llm=llm, retriever=retriever, memory=ConversationMemory()))
        # add should not raise — TextSplitter is pure Python
        await pipeline.add("Hello world. This is a test document.")

    @pytest.mark.asyncio
    async def test_auto_eval_non_blocking(self):
        llm = make_mock_llm()
        retriever = make_mock_retriever(chunks=["Context chunk 1."])
        memory = ConversationMemory()
        tracer = TokenTracer(model="gpt-4o-mini")
        pipeline = RAGPipeline(
            RAGConfig(
                llm=llm,
                retriever=retriever,
                memory=memory,
                tracer=tracer,
                auto_eval=True,
            )
        )

        started = asyncio.Event()
        release = asyncio.Event()

        async def fake_run_auto_eval(*, query, answer, contexts, call_id):
            started.set()
            await release.wait()

        pipeline._run_auto_eval = fake_run_auto_eval  # type: ignore[method-assign]

        answer = await asyncio.wait_for(pipeline.ask("What is this?"), timeout=0.5)
        assert answer == "Hello world"

        await asyncio.sleep(0)
        assert started.is_set(), "auto_eval task should be scheduled in the background"

        release.set()
        await pipeline.wait_for_auto_eval()

    @pytest.mark.asyncio
    async def test_auto_eval_records_quality_scores(self):
        llm = make_mock_llm()
        retriever = make_mock_retriever(chunks=["Context chunk 1."])
        memory = ConversationMemory()
        tracer = TokenTracer(model="gpt-4o-mini")
        pipeline = RAGPipeline(
            RAGConfig(
                llm=llm,
                retriever=retriever,
                memory=memory,
                tracer=tracer,
                auto_eval=True,
            )
        )

        async def fake_run_auto_eval(*, query, answer, contexts, call_id):
            tracer.record_quality(faithfulness=0.9, relevancy=0.8, call_id=call_id)

        pipeline._run_auto_eval = fake_run_auto_eval  # type: ignore[method-assign]

        await pipeline.ask("What is this?")
        await pipeline.wait_for_auto_eval()

        summary = tracer.summary()
        assert summary["avg_faithfulness"] == 0.9
        assert summary["avg_relevancy"] == 0.8

    @pytest.mark.asyncio
    async def test_auto_eval_disabled_by_default(self):
        llm = make_mock_llm()
        retriever = make_mock_retriever(chunks=["Context chunk 1."])
        memory = ConversationMemory()
        tracer = TokenTracer(model="gpt-4o-mini")
        pipeline = RAGPipeline(
            RAGConfig(
                llm=llm,
                retriever=retriever,
                memory=memory,
                tracer=tracer,
                auto_eval=False,
            )
        )

        called = False

        async def fake_run_auto_eval(*, query, answer, contexts, call_id):
            nonlocal called
            called = True

        pipeline._run_auto_eval = fake_run_auto_eval  # type: ignore[method-assign]

        await pipeline.ask("What is this?")
        await asyncio.sleep(0)

        assert not called
        assert tracer.summary()["avg_faithfulness"] == 0.0
