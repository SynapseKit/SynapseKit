"""Tests for InMemoryVectorStore."""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock

from synapsekit.retrieval.vectorstore import InMemoryVectorStore


def make_mock_embeddings(dim=4):
    """Return a mock SynapsekitEmbeddings that returns deterministic vectors."""
    mock = MagicMock()

    async def embed(texts):
        # Unique, normalised vectors based on text length
        vecs = []
        for i, t in enumerate(texts):
            v = np.zeros(dim, dtype=np.float32)
            v[i % dim] = 1.0
            vecs.append(v)
        return np.array(vecs, dtype=np.float32)

    async def embed_one(text):
        arr = await embed([text])
        return arr[0]

    mock.embed = embed
    mock.embed_one = embed_one
    return mock


@pytest.fixture
def store():
    return InMemoryVectorStore(make_mock_embeddings())


class TestInMemoryVectorStore:
    @pytest.mark.asyncio
    async def test_empty_search_returns_empty(self, store):
        results = await store.search("anything", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_add_and_search(self, store):
        await store.add(["text A", "text B", "text C"])
        assert len(store) == 3

        results = await store.search("text A", top_k=2)
        assert len(results) == 2
        assert all("text" in r and "score" in r and "metadata" in r for r in results)

    @pytest.mark.asyncio
    async def test_add_with_metadata(self, store):
        meta = [{"source": "doc1"}, {"source": "doc2"}]
        await store.add(["alpha", "beta"], metadata=meta)
        results = await store.search("alpha", top_k=2)
        sources = {r["metadata"].get("source") for r in results}
        assert "doc1" in sources or "doc2" in sources

    @pytest.mark.asyncio
    async def test_multiple_add_calls_accumulate(self, store):
        await store.add(["first"])
        await store.add(["second"])
        assert len(store) == 2

    @pytest.mark.asyncio
    async def test_top_k_respected(self, store):
        await store.add(["a", "b", "c", "d"])
        results = await store.search("query", top_k=2)
        assert len(results) == 2

    def test_save_raises_when_empty(self, store, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            store.save(str(tmp_path / "store.npz"))

    @pytest.mark.asyncio
    async def test_save_and_load_roundtrip(self, store, tmp_path):
        await store.add(["hello world"], metadata=[{"id": 1}])
        path = str(tmp_path / "store.npz")
        store.save(path)

        store2 = InMemoryVectorStore(make_mock_embeddings())
        store2.load(path)

        assert len(store2) == 1
        assert store2._texts[0] == "hello world"
        assert store2._metadata[0] == {"id": 1}
