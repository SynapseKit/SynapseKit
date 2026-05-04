"""Tests for FederatedRetriever."""

from __future__ import annotations

import asyncio

import pytest

from synapsekit.retrieval.federated import FederatedRetriever


class _MockRetriever:
    def __init__(self, results):
        self._results = results

    async def retrieve(self, query, top_k=5, metadata_filter=None):
        return self._results[:top_k]

    async def retrieve_with_scores(self, query, top_k=5, metadata_filter=None):
        return [
            {"text": t, "score": s, "metadata": {"id": i}}
            for i, (t, s) in enumerate(self._results[:top_k])
        ]


@pytest.mark.asyncio
async def test_federated_rrf_fusion_dedup():
    a = _MockRetriever([("doc a", 0.9), ("doc b", 0.8), ("doc c", 0.7)])
    b = _MockRetriever([("doc a", 0.95), ("doc d", 0.6)])

    retriever = FederatedRetriever(
        sources=[
            {"name": "a", "retriever": a},
            {"name": "b", "retriever": b},
        ],
        fusion="rrf",
        top_k=3,
    )

    results = await retriever.retrieve_with_scores("query")
    texts = [r["text"] for r in results]
    assert texts[0] == "doc a"
    assert len(texts) == 3
    assert len(set(texts)) == 3


@pytest.mark.asyncio
async def test_federated_interleave():
    a = _MockRetriever([("doc a", 0.9), ("doc b", 0.8)])
    b = _MockRetriever([("doc c", 0.7), ("doc d", 0.6)])

    retriever = FederatedRetriever(
        sources=[
            {"name": "a", "retriever": a},
            {"name": "b", "retriever": b},
        ],
        fusion="interleave",
        top_k=3,
    )

    results = await retriever.retrieve("query")
    assert results == ["doc a", "doc c", "doc b"]


@pytest.mark.asyncio
async def test_federated_timeout_returns_partial():
    class _SlowRetriever:
        async def retrieve_with_scores(self, query, top_k=5, metadata_filter=None):
            await asyncio.sleep(0.05)
            return [{"text": "late", "score": 0.1, "metadata": {}}]

    fast = _MockRetriever([("doc a", 0.9)])

    retriever = FederatedRetriever(
        sources=[
            {"name": "fast", "retriever": fast},
            {"name": "slow", "retriever": _SlowRetriever(), "timeout_ms": 1},
        ],
        fusion="rrf",
        top_k=2,
    )

    results = await retriever.retrieve("query")
    assert results == ["doc a"]
