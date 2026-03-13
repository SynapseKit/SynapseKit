"""Tests for v0.6.1 features: HITL, subgraphs, token streaming, self-query,
parent document, cross-encoder reranker, hybrid memory."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ------------------------------------------------------------------ #
# Import tests
# ------------------------------------------------------------------ #


def test_import_v061_features():
    from synapsekit import (
        CrossEncoderReranker,
        GraphInterrupt,
        HybridMemory,
        InterruptState,
        ParentDocumentRetriever,
        SelfQueryRetriever,
        llm_node,
        subgraph_node,
    )

    assert GraphInterrupt is not None
    assert InterruptState is not None
    assert llm_node is not None
    assert subgraph_node is not None
    assert SelfQueryRetriever is not None
    assert ParentDocumentRetriever is not None
    assert CrossEncoderReranker is not None
    assert HybridMemory is not None


# ------------------------------------------------------------------ #
# GraphInterrupt — Human-in-the-Loop
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_graph_interrupt_basic():
    """A node that raises GraphInterrupt should halt execution."""
    from synapsekit import GraphInterrupt, StateGraph

    call_count = 0

    async def step_one(state):
        nonlocal call_count
        call_count += 1
        return {"step": 1}

    async def review_node(state):
        raise GraphInterrupt(message="Needs review", data={"draft": "hello"})

    graph = StateGraph()
    graph.add_node("step_one", step_one)
    graph.add_node("review", review_node)
    graph.add_edge("step_one", "review")
    graph.set_entry_point("step_one")
    graph.set_finish_point("review")

    compiled = graph.compile()

    with pytest.raises(GraphInterrupt) as exc_info:
        await compiled.run({"input": "test"})

    assert exc_info.value.message == "Needs review"
    assert exc_info.value.data == {"draft": "hello"}
    assert call_count == 1


@pytest.mark.asyncio
async def test_graph_interrupt_with_checkpoint_and_resume():
    """Interrupt should checkpoint state; resume should continue with updates."""
    from synapsekit import GraphInterrupt, InMemoryCheckpointer, StateGraph

    async def step_one(state):
        return {"value": state.get("value", 0) + 1}

    interrupt_count = 0

    async def review_node(state):
        nonlocal interrupt_count
        interrupt_count += 1
        if not state.get("approved"):
            raise GraphInterrupt(message="Approve?", data={"value": state["value"]})
        return {"reviewed": True}

    graph = StateGraph()
    graph.add_node("step", step_one)
    graph.add_node("review", review_node)
    graph.add_edge("step", "review")
    graph.set_entry_point("step")
    graph.set_finish_point("review")

    compiled = graph.compile()
    cp = InMemoryCheckpointer()

    # First run — should interrupt
    with pytest.raises(GraphInterrupt):
        await compiled.run({"value": 0}, checkpointer=cp, graph_id="test-1")

    # Checkpoint should exist
    saved = cp.load("test-1")
    assert saved is not None

    # Resume with approval
    result = await compiled.resume("test-1", cp, updates={"approved": True})
    assert result["reviewed"] is True


@pytest.mark.asyncio
async def test_interrupt_state_repr():
    from synapsekit import InterruptState

    s = InterruptState(graph_id="g1", node="review", state={}, message="paused", data={}, step=2)
    assert "g1" in repr(s)
    assert "review" in repr(s)


# ------------------------------------------------------------------ #
# Subgraphs
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_subgraph_node_basic():
    """A subgraph nested in a parent graph should execute correctly."""
    from synapsekit import StateGraph, subgraph_node

    async def double(state):
        return {"value": state["value"] * 2}

    # Build subgraph
    sub = StateGraph()
    sub.add_node("double", double)
    sub.set_entry_point("double")
    sub.set_finish_point("double")
    compiled_sub = sub.compile()

    # Build parent graph
    async def prepare(state):
        return {"sub_input": state["input"] + 10}

    parent = StateGraph()
    parent.add_node("prepare", prepare)
    parent.add_node(
        "sub",
        subgraph_node(
            compiled_sub,
            input_mapping={"sub_input": "value"},
            output_mapping={"value": "result"},
        ),
    )
    parent.add_edge("prepare", "sub")
    parent.set_entry_point("prepare")
    parent.set_finish_point("sub")
    compiled_parent = parent.compile()

    result = await compiled_parent.run({"input": 5})
    # prepare: sub_input = 5+10 = 15
    # subgraph: value = 15*2 = 30
    # output mapping: result = 30
    assert result["result"] == 30


@pytest.mark.asyncio
async def test_subgraph_node_no_mapping():
    """Without mappings, subgraph gets full parent state."""
    from synapsekit import StateGraph, subgraph_node

    async def add_one(state):
        return {"count": state.get("count", 0) + 1}

    sub = StateGraph()
    sub.add_node("add", add_one)
    sub.set_entry_point("add")
    sub.set_finish_point("add")
    compiled_sub = sub.compile()

    parent = StateGraph()
    parent.add_node("sub", subgraph_node(compiled_sub))
    parent.set_entry_point("sub")
    parent.set_finish_point("sub")
    compiled_parent = parent.compile()

    result = await compiled_parent.run({"count": 5})
    assert result["count"] == 6


# ------------------------------------------------------------------ #
# Token-level streaming (llm_node + stream_tokens)
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_llm_node_non_streaming():
    from synapsekit import llm_node

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value="Hello world")

    fn = llm_node(mock_llm, input_key="prompt", output_key="response")
    result = await fn({"prompt": "Say hello"})
    assert result == {"response": "Hello world"}


@pytest.mark.asyncio
async def test_llm_node_streaming():
    from synapsekit import llm_node

    async def fake_stream(prompt):
        for token in ["Hello", " ", "world"]:
            yield token

    mock_llm = MagicMock()
    mock_llm.stream = fake_stream

    fn = llm_node(mock_llm, input_key="prompt", output_key="response", stream=True)
    result = await fn({"prompt": "Say hello"})
    assert "__stream__" in result
    assert result["__stream_key__"] == "response"


@pytest.mark.asyncio
async def test_stream_tokens():
    """stream_tokens() should yield token events from streaming nodes."""
    from synapsekit import StateGraph, llm_node

    async def fake_stream(prompt):
        for token in ["Hi", " ", "there"]:
            yield token

    mock_llm = MagicMock()
    mock_llm.stream = fake_stream

    graph = StateGraph()
    graph.add_node("llm", llm_node(mock_llm, stream=True))
    graph.set_entry_point("llm")
    graph.set_finish_point("llm")
    compiled = graph.compile()

    events = []
    async for event in compiled.stream_tokens({"input": "hello"}):
        events.append(event)

    token_events = [e for e in events if e["type"] == "token"]
    complete_events = [e for e in events if e["type"] == "node_complete"]

    assert len(token_events) == 3
    assert token_events[0]["token"] == "Hi"
    assert token_events[1]["token"] == " "
    assert token_events[2]["token"] == "there"
    assert len(complete_events) == 1
    assert complete_events[0]["state"]["output"] == "Hi there"


# ------------------------------------------------------------------ #
# SelfQueryRetriever
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_self_query_retriever():
    from synapsekit.retrieval.self_query import SelfQueryRetriever

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(
        return_value='{"query": "machine learning", "filters": {"year": "2024"}}'
    )

    mock_retriever = AsyncMock()
    mock_retriever.retrieve = AsyncMock(return_value=["doc1", "doc2"])

    sqr = SelfQueryRetriever(
        retriever=mock_retriever,
        llm=mock_llm,
        metadata_fields=["year", "author"],
    )

    results = await sqr.retrieve("ML papers from 2024")
    assert results == ["doc1", "doc2"]
    # Should have been called with extracted query and filters
    mock_retriever.retrieve.assert_called_once_with(
        "machine learning", top_k=5, metadata_filter={"year": "2024"}
    )


@pytest.mark.asyncio
async def test_self_query_retriever_no_filters():
    from synapsekit.retrieval.self_query import SelfQueryRetriever

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value='{"query": "neural networks", "filters": {}}')

    mock_retriever = AsyncMock()
    mock_retriever.retrieve = AsyncMock(return_value=["doc1"])

    sqr = SelfQueryRetriever(
        retriever=mock_retriever,
        llm=mock_llm,
        metadata_fields=["year"],
    )

    await sqr.retrieve("neural networks")
    mock_retriever.retrieve.assert_called_once_with(
        "neural networks", top_k=5, metadata_filter=None
    )


@pytest.mark.asyncio
async def test_self_query_retriever_invalid_json_fallback():
    from synapsekit.retrieval.self_query import SelfQueryRetriever

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value="not valid json")

    mock_retriever = AsyncMock()
    mock_retriever.retrieve = AsyncMock(return_value=["doc1"])

    sqr = SelfQueryRetriever(
        retriever=mock_retriever,
        llm=mock_llm,
        metadata_fields=["year"],
    )

    await sqr.retrieve("some query")
    # Should fallback to original question with no filters
    mock_retriever.retrieve.assert_called_once_with("some query", top_k=5, metadata_filter=None)


@pytest.mark.asyncio
async def test_self_query_retrieve_with_filters():
    from synapsekit.retrieval.self_query import SelfQueryRetriever

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value='{"query": "AI", "filters": {"author": "Smith"}}')

    mock_retriever = AsyncMock()
    mock_retriever.retrieve = AsyncMock(return_value=["doc1"])

    sqr = SelfQueryRetriever(
        retriever=mock_retriever,
        llm=mock_llm,
        metadata_fields=["author", "year"],
    )

    results, info = await sqr.retrieve_with_filters("AI by Smith")
    assert results == ["doc1"]
    assert info["query"] == "AI"
    assert info["filters"] == {"author": "Smith"}


# ------------------------------------------------------------------ #
# ParentDocumentRetriever
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_parent_document_retriever():
    from synapsekit.retrieval.parent_document import ParentDocumentRetriever

    mock_retriever = AsyncMock()
    mock_retriever.add = AsyncMock()

    pdr = ParentDocumentRetriever(retriever=mock_retriever, chunk_size=50, chunk_overlap=10)

    # Add documents
    docs = ["A" * 100, "B" * 30]  # First doc will be chunked, second won't
    await pdr.add_documents(docs)

    # Verify chunks were added
    assert mock_retriever.add.called
    call_args = mock_retriever.add.call_args
    chunks = call_args[0][0]
    metadata = call_args[0][1]
    # First doc: 100 chars / 50 chunk = ~3 chunks, second doc: 1 chunk
    assert len(chunks) >= 3
    assert all("_parent_id" in m for m in metadata)


def test_parent_document_chunking():
    from synapsekit.retrieval.parent_document import ParentDocumentRetriever

    mock_retriever = MagicMock()
    pdr = ParentDocumentRetriever(retriever=mock_retriever, chunk_size=10, chunk_overlap=3)

    chunks = pdr._chunk_text("Hello World, this is a test!")
    assert len(chunks) > 1
    assert all(len(c) <= 10 for c in chunks)


def test_parent_document_small_text():
    from synapsekit.retrieval.parent_document import ParentDocumentRetriever

    mock_retriever = MagicMock()
    pdr = ParentDocumentRetriever(retriever=mock_retriever, chunk_size=200)

    chunks = pdr._chunk_text("Short text")
    assert len(chunks) == 1
    assert chunks[0] == "Short text"


@pytest.mark.asyncio
async def test_parent_document_retrieve():
    from synapsekit.retrieval.parent_document import ParentDocumentRetriever

    mock_retriever = AsyncMock()
    mock_retriever.add = AsyncMock()

    pdr = ParentDocumentRetriever(retriever=mock_retriever, chunk_size=50)

    docs = ["Full document one content here.", "Full document two content here."]
    await pdr.add_documents(docs)

    # Get parent IDs from stored parents
    parent_ids = list(pdr._parents.keys())

    # Mock retrieve_with_scores to return chunks with parent metadata
    mock_retriever.retrieve_with_scores = AsyncMock(
        return_value=[
            {"text": "chunk1", "score": 0.9, "metadata": {"_parent_id": parent_ids[0]}},
            {"text": "chunk2", "score": 0.8, "metadata": {"_parent_id": parent_ids[1]}},
        ]
    )

    results = await pdr.retrieve("query", top_k=2)
    assert len(results) == 2
    assert results[0] == docs[0]
    assert results[1] == docs[1]


@pytest.mark.asyncio
async def test_parent_document_skips_empty():
    from synapsekit.retrieval.parent_document import ParentDocumentRetriever

    mock_retriever = AsyncMock()
    mock_retriever.add = AsyncMock()

    pdr = ParentDocumentRetriever(retriever=mock_retriever)
    await pdr.add_documents(["valid", "", "  ", "also valid"])

    # Only 2 valid docs should produce parents
    assert len(pdr._parents) == 2


# ------------------------------------------------------------------ #
# CrossEncoderReranker
# ------------------------------------------------------------------ #


def test_cross_encoder_init():
    from synapsekit.retrieval.cross_encoder import CrossEncoderReranker

    mock_retriever = MagicMock()
    reranker = CrossEncoderReranker(retriever=mock_retriever, fetch_k=30)
    assert reranker._fetch_k == 30
    assert reranker._model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_cross_encoder_import_error():
    from synapsekit.retrieval.cross_encoder import CrossEncoderReranker

    mock_retriever = MagicMock()
    reranker = CrossEncoderReranker(retriever=mock_retriever)

    with patch.dict("sys.modules", {"sentence_transformers": None}):
        with pytest.raises(ImportError, match="sentence-transformers"):
            reranker._cross_encoder = None  # Reset cache
            reranker._get_cross_encoder()


# ------------------------------------------------------------------ #
# HybridMemory
# ------------------------------------------------------------------ #


def test_hybrid_memory_basic():
    from synapsekit import HybridMemory

    mock_llm = MagicMock()
    mem = HybridMemory(llm=mock_llm, window=3)

    mem.add("user", "Hello")
    mem.add("assistant", "Hi there")
    assert len(mem) == 2
    assert mem.get_messages() == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]


def test_hybrid_memory_window_validation():
    from synapsekit import HybridMemory

    with pytest.raises(ValueError, match="window must be >= 1"):
        HybridMemory(llm=MagicMock(), window=0)


def test_hybrid_memory_recent_messages():
    from synapsekit import HybridMemory

    mem = HybridMemory(llm=MagicMock(), window=2)
    for i in range(10):
        mem.add("user", f"msg {i}")
        mem.add("assistant", f"reply {i}")

    recent = mem.get_recent_messages()
    # window=2 means max_messages = 2*2 = 4
    assert len(recent) == 4
    assert recent[0]["content"] == "msg 8"


@pytest.mark.asyncio
async def test_hybrid_memory_get_messages_with_summary_short():
    """When messages fit in window, no summary is generated."""
    from synapsekit import HybridMemory

    mock_llm = AsyncMock()
    mem = HybridMemory(llm=mock_llm, window=5)

    mem.add("user", "Hello")
    mem.add("assistant", "Hi")

    messages = await mem.get_messages_with_summary()
    assert len(messages) == 2
    mock_llm.generate.assert_not_called()


@pytest.mark.asyncio
async def test_hybrid_memory_get_messages_with_summary_long():
    """When messages exceed window, older ones are summarized."""
    from synapsekit import HybridMemory

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value="Summary of earlier conversation.")

    mem = HybridMemory(llm=mock_llm, window=2)
    for i in range(10):
        mem.add("user", f"msg {i}")
        mem.add("assistant", f"reply {i}")

    messages = await mem.get_messages_with_summary()
    # Should have summary + 4 recent messages (window=2, max_messages=4)
    assert messages[0]["role"] == "system"
    assert "Summary" in messages[0]["content"]
    assert len(messages) == 5  # 1 summary + 4 recent
    mock_llm.generate.assert_called_once()


@pytest.mark.asyncio
async def test_hybrid_memory_format_context():
    from synapsekit import HybridMemory

    mock_llm = AsyncMock()
    mem = HybridMemory(llm=mock_llm, window=5)
    mem.add("user", "Hello")
    mem.add("assistant", "Hi")

    ctx = await mem.format_context()
    assert "User: Hello" in ctx
    assert "Assistant: Hi" in ctx


def test_hybrid_memory_clear():
    from synapsekit import HybridMemory

    mem = HybridMemory(llm=MagicMock(), window=3)
    mem.add("user", "Hello")
    mem.add("assistant", "Hi")
    mem._summary = "old summary"

    mem.clear()
    assert len(mem) == 0
    assert mem.summary == ""


def test_hybrid_memory_summary_property():
    from synapsekit import HybridMemory

    mem = HybridMemory(llm=MagicMock(), window=3)
    assert mem.summary == ""
    mem._summary = "test"
    assert mem.summary == "test"


# ------------------------------------------------------------------ #
# Graph cycles (allow_cycles)
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_graph_with_cycles():
    """Graphs compiled with allow_cycles=True should support loops."""
    from synapsekit import StateGraph

    async def increment(state):
        return {"count": state.get("count", 0) + 1}

    def should_continue(state):
        return "loop" if state["count"] < 3 else "end"

    graph = StateGraph()
    graph.add_node("inc", increment)
    graph.set_entry_point("inc")
    graph.add_conditional_edge("inc", should_continue, {"loop": "inc", "end": "__end__"})

    compiled = graph.compile(allow_cycles=True, max_steps=10)
    result = await compiled.run({"count": 0})
    assert result["count"] == 3


@pytest.mark.asyncio
async def test_graph_cycles_max_steps():
    """Exceeding max_steps should raise GraphRuntimeError."""
    from synapsekit import GraphRuntimeError, StateGraph

    async def noop(state):
        return {"i": state.get("i", 0) + 1}

    def always_loop(state):
        return "loop"

    graph = StateGraph()
    graph.add_node("n", noop)
    graph.set_entry_point("n")
    graph.add_conditional_edge("n", always_loop, {"loop": "n"})

    compiled = graph.compile(allow_cycles=True, max_steps=5)
    with pytest.raises(GraphRuntimeError, match="exceeded _MAX_STEPS=5"):
        await compiled.run({"i": 0})
