"""Tests for Retriever — vector search + optional rerank."""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from synapsekit.retrieval.retriever import Retriever
from synapsekit.retrieval.vectorstore import InMemoryVectorStore


def make_mock_store(results=None):
    store = MagicMock(spec=InMemoryVectorStore)
    default = [
        {"text": "chunk A", "score": 0.9, "metadata": {}},
        {"text": "chunk B", "score": 0.7, "metadata": {}},
        {"text": "chunk C", "score": 0.5, "metadata": {}},
    ]
    store.search = AsyncMock(return_value=results if results is not None else default)
    return store


class TestRetriever:
    @pytest.mark.asyncio
    async def test_retrieve_returns_texts(self):
        store = make_mock_store()
        retriever = Retriever(store, rerank=False)
        texts = await retriever.retrieve("query", top_k=2)
        assert texts == ["chunk A", "chunk B"]

    @pytest.mark.asyncio
    async def test_retrieve_empty_store_returns_empty(self):
        store = make_mock_store(results=[])
        retriever = Retriever(store, rerank=False)
        texts = await retriever.retrieve("query")
        assert texts == []

    @pytest.mark.asyncio
    async def test_retrieve_with_scores(self):
        store = make_mock_store()
        retriever = Retriever(store, rerank=False)
        results = await retriever.retrieve_with_scores("query", top_k=2)
        assert len(results) == 2
        assert results[0]["score"] > results[1]["score"]

    @pytest.mark.asyncio
    async def test_rerank_applied(self):
        store = make_mock_store()
        retriever = Retriever(store, rerank=True)

        # Patch _bm25_rerank to control rerank order
        retriever._bm25_rerank = MagicMock(return_value=["chunk C", "chunk A"])

        texts = await retriever.retrieve("query", top_k=2)
        assert texts[0] == "chunk C"
        assert texts[1] == "chunk A"

    @pytest.mark.asyncio
    async def test_missing_rank_bm25_raises(self):
        store = make_mock_store()
        retriever = Retriever(store, rerank=True)
        with patch.dict("sys.modules", {"rank_bm25": None}):
            with pytest.raises(ImportError, match="rank-bm25"):
                retriever._bm25_rerank("query", ["a", "b"], 1)
