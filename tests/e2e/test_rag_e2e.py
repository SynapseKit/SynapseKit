"""End-to-end RAG pipeline tests.

These tests exercise the full RAG flow (add → retrieve → ask) using mocked LLMs.
No API keys are required; no network calls are made.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from synapsekit import RAG
from synapsekit.loaders import Document, StringLoader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rag(tokens=("Paris",)) -> RAG:
    """Create a RAG instance with all external calls mocked."""
    rag = RAG(model="gpt-4o-mini", api_key="sk-test")

    # Mock retriever
    rag._pipeline.config.retriever.retrieve = AsyncMock(
        return_value=["France is a country in Europe. Its capital is Paris."]
    )
    rag._pipeline.config.retriever._store.add = AsyncMock()

    # Mock LLM streaming
    async def _stream(messages, **kw):
        for tok in tokens:
            yield tok

    rag._pipeline.config.llm.stream_with_messages = _stream
    rag._pipeline.config.llm._input_tokens = 10
    rag._pipeline.config.llm._output_tokens = 5

    # Mock text splitter
    splitter = MagicMock()
    splitter.split = MagicMock(return_value=["chunk1", "chunk2"])
    rag._pipeline._splitter = splitter

    return rag


# ---------------------------------------------------------------------------
# 1. Add → ask (sync)
# ---------------------------------------------------------------------------


def test_rag_add_then_ask_sync():
    rag = _make_rag(tokens=["Paris"])
    rag.add("France is a country in Europe. Its capital is Paris.")
    answer = rag.ask_sync("What is the capital of France?")
    assert answer == "Paris"


# ---------------------------------------------------------------------------
# 2. Add → ask (async)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rag_add_async_then_ask():
    rag = _make_rag(tokens=["The", " capital", " is", " Paris", "."])
    await rag.add_async("France is a country in Europe. Its capital is Paris.")
    answer = await rag.ask("What is the capital of France?")
    assert "Paris" in answer


# ---------------------------------------------------------------------------
# 3. Stream returns tokens in order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rag_stream_order():
    expected = ["Token1", " Token2", " Token3"]
    rag = _make_rag(tokens=expected)
    rag.add("Some document.")

    collected = []
    async for tok in rag.stream("question?"):
        collected.append(tok)

    assert collected == expected


# ---------------------------------------------------------------------------
# 4. add_documents with Document objects
# ---------------------------------------------------------------------------


def test_rag_add_documents():
    rag = _make_rag()
    docs = [
        Document(text="First doc", metadata={"src": "a"}),
        Document(text="Second doc", metadata={"src": "b"}),
    ]
    rag.add_documents(docs)
    # splitter is called once per document
    assert rag._pipeline._splitter.split.call_count == 2


@pytest.mark.asyncio
async def test_rag_add_documents_async():
    rag = _make_rag()
    docs = [Document(text="Async doc", metadata={})]
    await rag.add_documents_async(docs)
    rag._pipeline._splitter.split.assert_called_once()


# ---------------------------------------------------------------------------
# 5. Loader → add_documents pipeline
# ---------------------------------------------------------------------------


def test_rag_with_string_loader():
    rag = _make_rag(tokens=["answer"])
    loader = StringLoader("Important knowledge about AI.")
    docs = loader.load()
    rag.add_documents(docs)
    answer = rag.ask_sync("What is this about?")
    assert answer == "answer"


# ---------------------------------------------------------------------------
# 6. Memory accumulates across calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rag_memory_accumulates():
    rag = _make_rag(tokens=["answer"])
    await rag.ask("First question?")
    await rag.ask("Second question?")
    # Memory should hold at least the user messages from both calls
    assert len(rag.memory) >= 2


# ---------------------------------------------------------------------------
# 7. Tracer records calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rag_tracer_records_usage():
    rag = _make_rag(tokens=["response"])
    await rag.ask("query?")
    summary = rag.tracer.summary()
    assert summary["calls"] >= 1
    assert summary["total_tokens"] > 0


# ---------------------------------------------------------------------------
# 8. Save/load vectorstore round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rag_save_load_round_trip(tmp_path):
    rag = _make_rag(tokens=["saved"])

    # Populate the vectorstore directly using the correct internal attribute names
    import numpy as np

    rag._vectorstore._texts = ["hello world"]
    rag._vectorstore._metadata = [{}]
    rag._vectorstore._vectors = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

    path = str(tmp_path / "store.npz")
    rag.save(path)

    # Load into a fresh RAG instance
    rag2 = _make_rag(tokens=["loaded"])
    rag2.load(path)

    assert rag2._vectorstore._texts == ["hello world"]


# ---------------------------------------------------------------------------
# 9. Retrieval_top_k is respected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rag_retrieval_top_k_passed_to_retriever():
    rag = RAG(model="gpt-4o-mini", api_key="sk-test", retrieval_top_k=3)

    retrieve_mock = AsyncMock(return_value=["ctx"])
    rag._pipeline.config.retriever.retrieve = retrieve_mock
    rag._pipeline.config.retriever._store.add = AsyncMock()

    async def _stream(messages, **kw):
        yield "ok"

    rag._pipeline.config.llm.stream_with_messages = _stream
    rag._pipeline.config.llm._input_tokens = 1
    rag._pipeline.config.llm._output_tokens = 1

    splitter = MagicMock()
    splitter.split = MagicMock(return_value=["chunk"])
    rag._pipeline._splitter = splitter

    await rag.ask("question?")
    retrieve_mock.assert_awaited_once()
    call_kwargs = retrieve_mock.call_args
    # top_k should be 3 (passed positionally or as kwarg)
    assert 3 in call_kwargs.args or call_kwargs.kwargs.get("top_k") == 3


# ---------------------------------------------------------------------------
# 10. Trace=False disables tracer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rag_trace_false_records_nothing():
    rag = RAG(model="gpt-4o-mini", api_key="sk-test", trace=False)

    rag._pipeline.config.retriever.retrieve = AsyncMock(return_value=["ctx"])
    rag._pipeline.config.retriever._store.add = AsyncMock()

    async def _stream(messages, **kw):
        yield "hi"

    rag._pipeline.config.llm.stream_with_messages = _stream
    rag._pipeline.config.llm._input_tokens = 1
    rag._pipeline.config.llm._output_tokens = 1
    splitter = MagicMock()
    splitter.split = MagicMock(return_value=["chunk"])
    rag._pipeline._splitter = splitter

    await rag.ask("test?")
    assert rag.tracer.summary()["calls"] == 0
