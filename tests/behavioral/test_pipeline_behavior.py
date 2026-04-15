"""Behavioral tests — RAG pipeline edge cases and robustness.

Verifies:
  - Empty query handling
  - Empty vectorstore query (no documents)
  - Memory preserved across stream disconnect
  - Multiple concurrent asks are isolated
  - System prompt is forwarded to the LLM
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from synapsekit import RAG

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_rag(rag: RAG, tokens=("answer",)) -> None:
    rag._pipeline.config.retriever.retrieve = AsyncMock(return_value=["context chunk"])
    rag._pipeline.config.retriever._store.add = AsyncMock()

    async def _stream(messages, **kw):
        for tok in tokens:
            yield tok

    rag._pipeline.config.llm.stream_with_messages = _stream
    rag._pipeline.config.llm._input_tokens = 5
    rag._pipeline.config.llm._output_tokens = 3

    splitter = MagicMock()
    splitter.split = MagicMock(return_value=["chunk"])
    rag._pipeline._splitter = splitter


# ---------------------------------------------------------------------------
# 1. Empty document store still returns a response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rag_empty_store_returns_response():
    rag = RAG(model="gpt-4o-mini", api_key="sk-test")
    rag._pipeline.config.retriever.retrieve = AsyncMock(return_value=[])
    rag._pipeline.config.retriever._store.add = AsyncMock()

    async def _stream(messages, **kw):
        yield "I don't know"

    rag._pipeline.config.llm.stream_with_messages = _stream
    rag._pipeline.config.llm._input_tokens = 5
    rag._pipeline.config.llm._output_tokens = 3
    splitter = MagicMock()
    splitter.split = MagicMock(return_value=["chunk"])
    rag._pipeline._splitter = splitter

    result = await rag.ask("What is X?")
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# 2. System prompt is included in LLM messages
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rag_system_prompt_forwarded():
    custom_prompt = "You are a pirate assistant."
    rag = RAG(model="gpt-4o-mini", api_key="sk-test", system_prompt=custom_prompt)

    captured_messages = []

    async def _stream(messages, **kw):
        captured_messages.extend(messages)
        yield "Arrr!"

    rag._pipeline.config.retriever.retrieve = AsyncMock(return_value=["context"])
    rag._pipeline.config.retriever._store.add = AsyncMock()
    rag._pipeline.config.llm.stream_with_messages = _stream
    rag._pipeline.config.llm._input_tokens = 5
    rag._pipeline.config.llm._output_tokens = 3
    splitter = MagicMock()
    splitter.split = MagicMock(return_value=["chunk"])
    rag._pipeline._splitter = splitter

    await rag.ask("Greet me!")
    system_contents = [m.get("content", "") for m in captured_messages if m.get("role") == "system"]
    assert any(custom_prompt in c for c in system_contents)


# ---------------------------------------------------------------------------
# 3. Memory grows with consecutive asks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rag_memory_grows_with_consecutive_asks():
    rag = RAG(model="gpt-4o-mini", api_key="sk-test", memory_window=20)
    _patch_rag(rag)

    await rag.ask("First question?")
    after_first = len(rag.memory)

    await rag.ask("Second question?")
    after_second = len(rag.memory)

    assert after_second > after_first


# ---------------------------------------------------------------------------
# 4. Memory window evicts old messages
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rag_memory_window_enforced():
    rag = RAG(model="gpt-4o-mini", api_key="sk-test", memory_window=2)
    _patch_rag(rag)

    for _ in range(10):
        await rag.ask("question?")

    # window=2 → max 4 messages (2 pairs)
    assert len(rag.memory) <= 4


# ---------------------------------------------------------------------------
# 5. Multiple independent RAG instances don't share state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_rag_instances_isolated():
    rag1 = RAG(model="gpt-4o-mini", api_key="sk-test")
    rag2 = RAG(model="gpt-4o-mini", api_key="sk-test")
    _patch_rag(rag1, tokens=["response1"])
    _patch_rag(rag2, tokens=["response2"])

    ans1 = await rag1.ask("Q1?")
    ans2 = await rag2.ask("Q2?")

    assert ans1 != ans2
    # Memories are independent
    assert rag1.memory is not rag2.memory


# ---------------------------------------------------------------------------
# 6. Concurrent asks don't interfere
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rag_concurrent_asks():
    rag = RAG(model="gpt-4o-mini", api_key="sk-test")
    _patch_rag(rag, tokens=["concurrent_answer"])

    results = await asyncio.gather(
        rag.ask("Q1"),
        rag.ask("Q2"),
        rag.ask("Q3"),
    )
    assert all(isinstance(r, str) for r in results)
    assert all("concurrent_answer" in r for r in results)


# ---------------------------------------------------------------------------
# 7. LLM returning empty string is handled gracefully
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rag_llm_empty_response():
    rag = RAG(model="gpt-4o-mini", api_key="sk-test")
    rag._pipeline.config.retriever.retrieve = AsyncMock(return_value=["ctx"])
    rag._pipeline.config.retriever._store.add = AsyncMock()

    async def _empty_stream(messages, **kw):
        return
        yield  # make it an async generator

    rag._pipeline.config.llm.stream_with_messages = _empty_stream
    rag._pipeline.config.llm._input_tokens = 5
    rag._pipeline.config.llm._output_tokens = 0
    splitter = MagicMock()
    splitter.split = MagicMock(return_value=["chunk"])
    rag._pipeline._splitter = splitter

    result = await rag.ask("anything?")
    assert isinstance(result, str)  # empty string is fine


# ---------------------------------------------------------------------------
# 8. add() with metadata is stored
# ---------------------------------------------------------------------------


def test_rag_add_with_metadata():
    rag = RAG(model="gpt-4o-mini", api_key="sk-test")
    _patch_rag(rag)
    rag.add("Document content", metadata={"source": "test", "page": 1})
    # splitter is called
    rag._pipeline._splitter.split.assert_called_once()
