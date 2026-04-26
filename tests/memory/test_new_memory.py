"""Tests for VectorConversationMemory, ReadOnlySharedMemory, KnowledgeGraphMemory."""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# VectorConversationMemory
# ---------------------------------------------------------------------------


class TestVectorConversationMemory:
    def _make_embeddings(self, dim: int = 4):
        """Return a mock SynapsekitEmbeddings that returns unit vectors."""
        emb = AsyncMock()
        emb.embed = AsyncMock(
            side_effect=lambda texts: np.ones((len(texts), dim), dtype=np.float32)
            / np.sqrt(dim)
        )
        emb.embed_one = AsyncMock(return_value=np.ones(dim, dtype=np.float32) / np.sqrt(dim))
        return emb

    @pytest.mark.asyncio
    async def test_add_message_and_get_context(self):
        from synapsekit.memory.vector_memory import VectorConversationMemory

        emb = self._make_embeddings()
        mem = VectorConversationMemory(embedding_backend=emb)
        await mem.add_message("user", "Hello there")
        await mem.add_message("assistant", "Hi!")

        ctx = await mem.get_context("Hello")
        assert isinstance(ctx, list)
        assert len(ctx) > 0
        assert all("role" in m and "content" in m for m in ctx)

    @pytest.mark.asyncio
    async def test_get_context_respects_k(self):
        from synapsekit.memory.vector_memory import VectorConversationMemory

        emb = self._make_embeddings()
        mem = VectorConversationMemory(embedding_backend=emb, k=2)
        for i in range(5):
            await mem.add_message("user", f"message {i}")
        ctx = await mem.get_context("query", k=2)
        assert len(ctx) <= 2

    @pytest.mark.asyncio
    async def test_clear_resets_state(self):
        from synapsekit.memory.vector_memory import VectorConversationMemory

        emb = self._make_embeddings()
        mem = VectorConversationMemory(embedding_backend=emb)
        await mem.add_message("user", "hello")
        mem.clear()
        assert len(mem) == 0
        assert mem.messages == []

    @pytest.mark.asyncio
    async def test_messages_property_returns_copy(self):
        from synapsekit.memory.vector_memory import VectorConversationMemory

        emb = self._make_embeddings()
        mem = VectorConversationMemory(embedding_backend=emb)
        await mem.add_message("user", "test")
        msgs = mem.messages
        msgs.append({"role": "injected", "content": "bad"})
        assert len(mem) == 1

    @pytest.mark.asyncio
    async def test_get_context_empty_store(self):
        from synapsekit.memory.vector_memory import VectorConversationMemory

        emb = self._make_embeddings()
        mem = VectorConversationMemory(embedding_backend=emb)
        ctx = await mem.get_context("query")
        assert ctx == []

    def test_add_message_is_coroutine(self):
        from synapsekit.memory.vector_memory import VectorConversationMemory

        assert inspect.iscoroutinefunction(VectorConversationMemory.add_message)

    def test_get_context_is_coroutine(self):
        from synapsekit.memory.vector_memory import VectorConversationMemory

        assert inspect.iscoroutinefunction(VectorConversationMemory.get_context)


# ---------------------------------------------------------------------------
# ReadOnlySharedMemory
# ---------------------------------------------------------------------------


class TestReadOnlySharedMemory:
    def _make_mem_with_messages(self, msgs=None):
        m = MagicMock()
        m.messages = msgs or [{"role": "user", "content": "hello"}]
        return m

    def test_messages_proxied(self):
        from synapsekit.memory.readonly_shared_memory import ReadOnlySharedMemory

        inner = self._make_mem_with_messages()
        ro = ReadOnlySharedMemory(memory=inner)
        assert ro.messages == inner.messages

    def test_add_message_raises_permission_error(self):
        from synapsekit.memory.readonly_shared_memory import ReadOnlySharedMemory

        inner = self._make_mem_with_messages()
        ro = ReadOnlySharedMemory(memory=inner)
        with pytest.raises(PermissionError, match="read-only"):
            ro.add_message("user", "test")

    @pytest.mark.asyncio
    async def test_get_context_proxied_sync(self):
        from synapsekit.memory.readonly_shared_memory import ReadOnlySharedMemory

        inner = MagicMock()
        inner.messages = [{"role": "user", "content": "hi"}]
        inner.get_context = MagicMock(return_value=[{"role": "user", "content": "hi"}])

        ro = ReadOnlySharedMemory(memory=inner)
        ctx = await ro.get_context("query")
        assert ctx == [{"role": "user", "content": "hi"}]

    @pytest.mark.asyncio
    async def test_get_context_proxied_async(self):
        from synapsekit.memory.readonly_shared_memory import ReadOnlySharedMemory

        inner = MagicMock()
        inner.messages = [{"role": "user", "content": "hi"}]
        inner.get_context = AsyncMock(return_value=[{"role": "user", "content": "hi"}])

        ro = ReadOnlySharedMemory(memory=inner)
        ctx = await ro.get_context("query")
        assert ctx == [{"role": "user", "content": "hi"}]

    @pytest.mark.asyncio
    async def test_get_context_fallback_to_messages(self):
        from synapsekit.memory.readonly_shared_memory import ReadOnlySharedMemory

        inner = MagicMock(spec=["messages"])
        inner.messages = [{"role": "user", "content": "fallback"}]

        ro = ReadOnlySharedMemory(memory=inner)
        ctx = await ro.get_context()
        assert ctx == inner.messages

    def test_messages_returns_copy(self):
        from synapsekit.memory.readonly_shared_memory import ReadOnlySharedMemory

        inner = self._make_mem_with_messages()
        ro = ReadOnlySharedMemory(memory=inner)
        msgs = ro.messages
        msgs.append({"role": "injected", "content": "bad"})
        assert len(ro.messages) == len(inner.messages)


# ---------------------------------------------------------------------------
# KnowledgeGraphMemory
# ---------------------------------------------------------------------------


class TestKnowledgeGraphMemory:
    def _make_llm(self, response: str = '[{"s":"Alice","p":"works_at","o":"Acme"}]'):
        llm = AsyncMock()
        llm.generate = AsyncMock(return_value=response)
        return llm

    @pytest.mark.asyncio
    async def test_add_message_extracts_triplets(self):
        from synapsekit.memory.knowledge_graph_memory import KnowledgeGraphMemory

        llm = self._make_llm()
        mem = KnowledgeGraphMemory(llm=llm)
        await mem.add_message("user", "Alice works at Acme.")
        assert len(mem.triplets) == 1
        assert mem.triplets[0] == ("Alice", "works_at", "Acme")

    @pytest.mark.asyncio
    async def test_graph_updated(self):
        from synapsekit.memory.knowledge_graph_memory import KnowledgeGraphMemory

        llm = self._make_llm()
        mem = KnowledgeGraphMemory(llm=llm)
        await mem.add_message("user", "Alice works at Acme.")
        assert "Alice" in mem.graph
        assert ("works_at", "Acme") in mem.graph["Alice"]

    @pytest.mark.asyncio
    async def test_get_context_returns_relevant_triplets(self):
        from synapsekit.memory.knowledge_graph_memory import KnowledgeGraphMemory

        llm = self._make_llm()
        mem = KnowledgeGraphMemory(llm=llm)
        await mem.add_message("user", "Alice works at Acme.")
        ctx = await mem.get_context("Where does Alice work?")
        assert isinstance(ctx, list)
        assert len(ctx) == 1
        assert "Alice" in ctx[0]["content"]

    @pytest.mark.asyncio
    async def test_graceful_parse_error(self):
        from synapsekit.memory.knowledge_graph_memory import KnowledgeGraphMemory

        llm = AsyncMock()
        llm.generate = AsyncMock(return_value="not valid json at all")
        mem = KnowledgeGraphMemory(llm=llm)
        # Should not raise
        await mem.add_message("user", "Some message.")
        assert len(mem.triplets) == 0

    @pytest.mark.asyncio
    async def test_clear(self):
        from synapsekit.memory.knowledge_graph_memory import KnowledgeGraphMemory

        llm = self._make_llm()
        mem = KnowledgeGraphMemory(llm=llm)
        await mem.add_message("user", "Alice works at Acme.")
        mem.clear()
        assert len(mem.triplets) == 0
        assert len(mem.graph) == 0
        assert len(mem) == 0

    @pytest.mark.asyncio
    async def test_max_triplets_eviction(self):
        from synapsekit.memory.knowledge_graph_memory import KnowledgeGraphMemory

        llm = AsyncMock()
        call_count = 0

        async def gen(prompt):
            nonlocal call_count
            call_count += 1
            return f'[{{"s":"E{call_count}","p":"rel","o":"T{call_count}"}}]'

        llm.generate = gen
        mem = KnowledgeGraphMemory(llm=llm, max_triplets=3)
        for i in range(5):
            await mem.add_message("user", f"msg {i}")
        assert len(mem.triplets) <= 3

    @pytest.mark.asyncio
    async def test_add_message_is_coroutine(self):
        from synapsekit.memory.knowledge_graph_memory import KnowledgeGraphMemory

        assert inspect.iscoroutinefunction(KnowledgeGraphMemory.add_message)

    @pytest.mark.asyncio
    async def test_get_context_fallback_to_all_triplets(self):
        """When no triplets match query, return all triplets."""
        from synapsekit.memory.knowledge_graph_memory import KnowledgeGraphMemory

        llm = self._make_llm()
        mem = KnowledgeGraphMemory(llm=llm)
        await mem.add_message("user", "Alice works at Acme.")
        ctx = await mem.get_context("something unrelated xyz")
        assert len(ctx) == 1  # Falls back to all triplets

    @pytest.mark.asyncio
    async def test_return_triplets_in_context_false(self):
        from synapsekit.memory.knowledge_graph_memory import KnowledgeGraphMemory

        llm = self._make_llm()
        mem = KnowledgeGraphMemory(llm=llm, return_triplets_in_context=False)
        await mem.add_message("user", "Alice works at Acme.")
        ctx = await mem.get_context("Alice")
        # Should return raw messages instead of triplet context
        assert any(m["role"] == "user" for m in ctx)
