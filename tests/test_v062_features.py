"""Tests for v0.6.2 features: CRAG, query decomposition, contextual compression,
ensemble retrieval, SQLite memory, summary buffer memory, human input tool, Wikipedia tool."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

# ------------------------------------------------------------------ #
# Import tests
# ------------------------------------------------------------------ #


def test_import_v062_features():
    from synapsekit import (
        ContextualCompressionRetriever,
        CRAGRetriever,
        EnsembleRetriever,
        HumanInputTool,
        QueryDecompositionRetriever,
        SQLiteConversationMemory,
        SummaryBufferMemory,
        WikipediaTool,
    )

    assert CRAGRetriever is not None
    assert QueryDecompositionRetriever is not None
    assert ContextualCompressionRetriever is not None
    assert EnsembleRetriever is not None
    assert SQLiteConversationMemory is not None
    assert SummaryBufferMemory is not None
    assert HumanInputTool is not None
    assert WikipediaTool is not None


# ------------------------------------------------------------------ #
# CRAG Retriever
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_crag_retriever_all_relevant():
    from synapsekit import CRAGRetriever

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value="relevant")

    mock_retriever = AsyncMock()
    mock_retriever.retrieve = AsyncMock(return_value=["doc1", "doc2", "doc3"])

    crag = CRAGRetriever(retriever=mock_retriever, llm=mock_llm)
    results = await crag.retrieve("test query", top_k=3)
    assert results == ["doc1", "doc2", "doc3"]


@pytest.mark.asyncio
async def test_crag_retriever_filters_irrelevant():
    from synapsekit import CRAGRetriever

    async def grade_response(prompt):
        # doc2 is irrelevant, others are relevant
        if "doc2" in prompt:
            return "irrelevant"
        return "relevant"

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(side_effect=grade_response)

    mock_retriever = AsyncMock()
    mock_retriever.retrieve = AsyncMock(return_value=["doc1", "doc2", "doc3"])

    crag = CRAGRetriever(retriever=mock_retriever, llm=mock_llm, max_retries=0)
    results = await crag.retrieve("test", top_k=5)
    assert "doc1" in results
    assert "doc3" in results
    assert "doc2" not in results


@pytest.mark.asyncio
async def test_crag_retriever_with_grades():
    from synapsekit import CRAGRetriever

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value="relevant")

    mock_retriever = AsyncMock()
    mock_retriever.retrieve = AsyncMock(return_value=["doc1"])

    crag = CRAGRetriever(retriever=mock_retriever, llm=mock_llm)
    results, meta = await crag.retrieve_with_grades("test")
    assert len(results) == 1
    assert meta["total_candidates"] == 1
    assert meta["relevant_count"] == 1
    assert meta["grades"][0]["relevant"] is True


@pytest.mark.asyncio
async def test_crag_retriever_rewrites_on_poor_results():
    from synapsekit import CRAGRetriever

    async def mock_generate(prompt):
        if "Rewrite" in prompt:
            return "rewritten query"
        return "irrelevant"

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(side_effect=mock_generate)

    mock_retriever = AsyncMock()
    mock_retriever.retrieve = AsyncMock(return_value=["doc1"])

    crag = CRAGRetriever(
        retriever=mock_retriever, llm=mock_llm, relevance_threshold=1.0, max_retries=1
    )
    await crag.retrieve("original query")
    # Should have called retrieve twice (original + rewrite)
    assert mock_retriever.retrieve.call_count == 2


# ------------------------------------------------------------------ #
# Query Decomposition Retriever
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_query_decomposition_retriever():
    from synapsekit import QueryDecompositionRetriever

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value="Sub question 1\nSub question 2")

    call_results = {
        "original": ["doc1", "doc2"],
        "Sub question 1": ["doc2", "doc3"],
        "Sub question 2": ["doc4"],
    }

    async def mock_retrieve(query, top_k=5, metadata_filter=None):
        return call_results.get(query, [])

    mock_retriever = AsyncMock()
    mock_retriever.retrieve = AsyncMock(side_effect=mock_retrieve)

    qdr = QueryDecompositionRetriever(retriever=mock_retriever, llm=mock_llm, num_sub_queries=2)
    results = await qdr.retrieve("original", top_k=10)
    # Should deduplicate: doc1, doc2, doc3, doc4
    assert len(results) == 4
    assert "doc1" in results


@pytest.mark.asyncio
async def test_query_decomposition_with_sub_queries():
    from synapsekit import QueryDecompositionRetriever

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value="Sub 1\nSub 2")

    mock_retriever = AsyncMock()
    mock_retriever.retrieve = AsyncMock(return_value=["doc1"])

    qdr = QueryDecompositionRetriever(retriever=mock_retriever, llm=mock_llm)
    _results, sub_queries = await qdr.retrieve_with_sub_queries("main query")
    assert "main query" in sub_queries
    assert "Sub 1" in sub_queries


# ------------------------------------------------------------------ #
# Contextual Compression Retriever
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_contextual_compression_retriever():
    from synapsekit import ContextualCompressionRetriever

    async def compress_response(prompt):
        if "doc1" in prompt:
            return "Relevant excerpt from doc1"
        return "NOT_RELEVANT"

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(side_effect=compress_response)

    mock_retriever = AsyncMock()
    mock_retriever.retrieve = AsyncMock(return_value=["doc1 full text", "doc2 full text"])

    ccr = ContextualCompressionRetriever(retriever=mock_retriever, llm=mock_llm)
    results = await ccr.retrieve("query", top_k=5)
    assert len(results) == 1
    assert results[0] == "Relevant excerpt from doc1"


@pytest.mark.asyncio
async def test_contextual_compression_all_relevant():
    from synapsekit import ContextualCompressionRetriever

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value="compressed text")

    mock_retriever = AsyncMock()
    mock_retriever.retrieve = AsyncMock(return_value=["doc1", "doc2"])

    ccr = ContextualCompressionRetriever(retriever=mock_retriever, llm=mock_llm)
    results = await ccr.retrieve("query")
    assert len(results) == 2


# ------------------------------------------------------------------ #
# Ensemble Retriever
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_ensemble_retriever():
    from synapsekit import EnsembleRetriever

    retriever_a = AsyncMock()
    retriever_a.retrieve = AsyncMock(return_value=["doc1", "doc2", "doc3"])

    retriever_b = AsyncMock()
    retriever_b.retrieve = AsyncMock(return_value=["doc2", "doc4", "doc1"])

    ensemble = EnsembleRetriever(retrievers=[retriever_a, retriever_b])
    results = await ensemble.retrieve("query", top_k=3)
    assert len(results) == 3
    # doc1 and doc2 appear in both, should rank higher
    assert "doc1" in results
    assert "doc2" in results


@pytest.mark.asyncio
async def test_ensemble_retriever_with_weights():
    from synapsekit import EnsembleRetriever

    retriever_a = AsyncMock()
    retriever_a.retrieve = AsyncMock(return_value=["docA"])

    retriever_b = AsyncMock()
    retriever_b.retrieve = AsyncMock(return_value=["docB"])

    # Weight B much higher
    ensemble = EnsembleRetriever(retrievers=[retriever_a, retriever_b], weights=[0.1, 10.0])
    results = await ensemble.retrieve("query", top_k=2)
    assert results[0] == "docB"  # Higher weight should rank first


def test_ensemble_retriever_validation():
    from synapsekit import EnsembleRetriever

    with pytest.raises(ValueError, match="At least one"):
        EnsembleRetriever(retrievers=[])

    with pytest.raises(ValueError, match="same length"):
        EnsembleRetriever(retrievers=[MagicMock()], weights=[1.0, 2.0])


# ------------------------------------------------------------------ #
# SQLite Conversation Memory
# ------------------------------------------------------------------ #


def test_sqlite_memory_basic():
    from synapsekit import SQLiteConversationMemory

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        mem = SQLiteConversationMemory(db_path=db_path, conversation_id="test")
        mem.add("user", "Hello")
        mem.add("assistant", "Hi there")
        assert len(mem) == 2

        messages = mem.get_messages()
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "content": "Hi there"}
        mem.close()
    finally:
        os.unlink(db_path)


def test_sqlite_memory_persistence():
    from synapsekit import SQLiteConversationMemory

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        # Write
        mem1 = SQLiteConversationMemory(db_path=db_path, conversation_id="test")
        mem1.add("user", "Hello")
        mem1.close()

        # Read from new instance
        mem2 = SQLiteConversationMemory(db_path=db_path, conversation_id="test")
        assert len(mem2) == 1
        assert mem2.get_messages()[0]["content"] == "Hello"
        mem2.close()
    finally:
        os.unlink(db_path)


def test_sqlite_memory_multiple_conversations():
    from synapsekit import SQLiteConversationMemory

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        mem_a = SQLiteConversationMemory(db_path=db_path, conversation_id="conv-a")
        mem_b = SQLiteConversationMemory(db_path=db_path, conversation_id="conv-b")

        mem_a.add("user", "Hello A")
        mem_b.add("user", "Hello B")

        assert len(mem_a) == 1
        assert len(mem_b) == 1
        assert mem_a.get_messages()[0]["content"] == "Hello A"
        assert mem_b.get_messages()[0]["content"] == "Hello B"

        convos = mem_a.list_conversations()
        assert "conv-a" in convos
        assert "conv-b" in convos

        mem_a.close()
        mem_b.close()
    finally:
        os.unlink(db_path)


def test_sqlite_memory_clear():
    from synapsekit import SQLiteConversationMemory

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        mem = SQLiteConversationMemory(db_path=db_path)
        mem.add("user", "Hello")
        assert len(mem) == 1
        mem.clear()
        assert len(mem) == 0
        mem.close()
    finally:
        os.unlink(db_path)


def test_sqlite_memory_format_context():
    from synapsekit import SQLiteConversationMemory

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        mem = SQLiteConversationMemory(db_path=db_path)
        mem.add("user", "Hello")
        mem.add("assistant", "Hi")
        ctx = mem.format_context()
        assert "User: Hello" in ctx
        assert "Assistant: Hi" in ctx
        mem.close()
    finally:
        os.unlink(db_path)


def test_sqlite_memory_window():
    from synapsekit import SQLiteConversationMemory

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        mem = SQLiteConversationMemory(db_path=db_path, window=2)
        for i in range(10):
            mem.add("user", f"msg {i}")
            mem.add("assistant", f"reply {i}")
        # window=2 -> max 4 messages
        assert len(mem) == 4
        messages = mem.get_messages()
        assert messages[0]["content"] == "msg 8"
        mem.close()
    finally:
        os.unlink(db_path)


# ------------------------------------------------------------------ #
# Summary Buffer Memory
# ------------------------------------------------------------------ #


def test_summary_buffer_memory_basic():
    from synapsekit import SummaryBufferMemory

    mem = SummaryBufferMemory(llm=MagicMock(), max_tokens=2000)
    mem.add("user", "Hello")
    mem.add("assistant", "Hi")
    assert len(mem) == 2


def test_summary_buffer_memory_validation():
    from synapsekit import SummaryBufferMemory

    with pytest.raises(ValueError, match="max_tokens must be >= 100"):
        SummaryBufferMemory(llm=MagicMock(), max_tokens=50)


@pytest.mark.asyncio
async def test_summary_buffer_no_summarization_when_under_limit():
    from synapsekit import SummaryBufferMemory

    mock_llm = AsyncMock()
    mem = SummaryBufferMemory(llm=mock_llm, max_tokens=2000)
    mem.add("user", "Hello")
    mem.add("assistant", "Hi")

    messages = await mem.get_messages()
    assert len(messages) == 2
    mock_llm.generate.assert_not_called()


@pytest.mark.asyncio
async def test_summary_buffer_summarizes_when_over_limit():
    from synapsekit import SummaryBufferMemory

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value="Summary of conversation.")

    # Very low token limit to trigger summarization
    mem = SummaryBufferMemory(llm=mock_llm, max_tokens=100, chars_per_token=1)
    for i in range(20):
        mem.add("user", f"This is a longer message number {i} with some content")
        mem.add("assistant", f"This is a reply to message number {i}")

    messages = await mem.get_messages()
    # Should have a summary system message
    assert any(m["role"] == "system" for m in messages)
    assert mem.summary != ""


def test_summary_buffer_format_context():
    from synapsekit import SummaryBufferMemory

    mem = SummaryBufferMemory(llm=MagicMock())
    mem.add("user", "Hello")
    mem._summary = "Earlier discussion about greetings"
    ctx = mem.format_context()
    assert "Summary:" in ctx
    assert "User: Hello" in ctx


def test_summary_buffer_clear():
    from synapsekit import SummaryBufferMemory

    mem = SummaryBufferMemory(llm=MagicMock())
    mem.add("user", "Hello")
    mem._summary = "old"
    mem.clear()
    assert len(mem) == 0
    assert mem.summary == ""


@pytest.mark.asyncio
async def test_summary_buffer_preserves_history_on_summarizer_failure():
    """If the summarizer LLM fails, messages must not be silently dropped."""
    from synapsekit import SummaryBufferMemory

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(side_effect=RuntimeError("rate limit"))

    mem = SummaryBufferMemory(llm=mock_llm, max_tokens=100, chars_per_token=1)
    # Fill the buffer above max_tokens so summarization triggers
    for i in range(20):
        mem.add("user", f"This is a longer message number {i} with some content")
        mem.add("assistant", f"This is a reply to message number {i}")

    initial_len = len(mem)

    with pytest.raises(RuntimeError, match="rate limit"):
        await mem.get_messages()

    # Invariant: on LLM failure, message buffer is unchanged AND summary is unchanged
    assert len(mem) == initial_len, (
        f"Messages were silently dropped on LLM failure: {initial_len} -> {len(mem)}"
    )
    assert mem.summary == "", "Summary should remain empty when summarizer failed"


# ------------------------------------------------------------------ #
# Human Input Tool
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_human_input_tool_custom_fn():
    from synapsekit import HumanInputTool

    tool = HumanInputTool(input_fn=lambda q: f"answer to: {q}")
    result = await tool.run(question="What is your name?")
    assert not result.is_error
    assert "answer to: What is your name?" in result.output


@pytest.mark.asyncio
async def test_human_input_tool_async_fn():
    from synapsekit import HumanInputTool

    async def async_input(q):
        return f"async answer to: {q}"

    tool = HumanInputTool(input_fn=async_input)
    result = await tool.run(question="Test?")
    assert not result.is_error
    assert "async answer" in result.output


@pytest.mark.asyncio
async def test_human_input_tool_no_question():
    from synapsekit import HumanInputTool

    tool = HumanInputTool(input_fn=lambda q: "response")
    result = await tool.run()
    assert result.is_error
    assert "No question" in result.error


def test_human_input_tool_schema():
    from synapsekit import HumanInputTool

    tool = HumanInputTool()
    assert tool.name == "human_input"
    assert "question" in tool.parameters["properties"]


# ------------------------------------------------------------------ #
# Wikipedia Tool
# ------------------------------------------------------------------ #


def test_wikipedia_tool_schema():
    from synapsekit import WikipediaTool

    tool = WikipediaTool()
    assert tool.name == "wikipedia"
    assert "query" in tool.parameters["properties"]


@pytest.mark.asyncio
async def test_wikipedia_tool_no_query():
    from synapsekit import WikipediaTool

    tool = WikipediaTool()
    result = await tool.run()
    assert result.is_error
    assert "No search query" in result.error
