"""Tests for the Reranker ABC and the Voyage/Jina/mixedbread rerankers."""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

respx = pytest.importorskip("respx")


def _mock_retriever(texts: list[str]) -> AsyncMock:
    retriever = AsyncMock()
    retriever.retrieve.return_value = texts
    return retriever


def _sorted_payload(items: list[tuple[int, float]]) -> dict:
    return {"data": [{"index": i, "relevance_score": s} for i, s in items]}


# --------------------------------------------------------------------------- #
# Reranker ABC
# --------------------------------------------------------------------------- #


def test_reranker_abc_requires_retrieve_and_retrieve_with_scores():
    from synapsekit.retrieval.reranker import Reranker

    assert Reranker.__abstractmethods__ == frozenset({"retrieve", "retrieve_with_scores"})


def test_cohere_and_cross_encoder_subclass_reranker():
    from synapsekit import CohereReranker, CrossEncoderReranker
    from synapsekit.retrieval.reranker import Reranker

    assert issubclass(CohereReranker, Reranker)
    assert issubclass(CrossEncoderReranker, Reranker)


# --------------------------------------------------------------------------- #
# VoyageReranker
# --------------------------------------------------------------------------- #


@respx.mock
def test_voyage_reranker_request_and_reorder():
    retriever = _mock_retriever(["doc1", "doc2", "doc3"])
    respx.post("https://api.voyageai.com/v1/rerank").mock(
        return_value=httpx.Response(200, json=_sorted_payload([(2, 0.9), (0, 0.7)]))
    )
    from synapsekit.retrieval.voyage_reranker import VoyageReranker

    reranker = VoyageReranker(retriever=retriever, api_key="test-key")
    results = reranker._call("query", ["doc1", "doc2", "doc3"], top_n=2)
    assert results == [
        {"text": "doc3", "relevance_score": 0.9},
        {"text": "doc1", "relevance_score": 0.7},
    ]


@pytest.mark.asyncio
@respx.mock
async def test_voyage_retrieve_async():
    retriever = _mock_retriever(["doc1", "doc2", "doc3"])
    respx.post("https://api.voyageai.com/v1/rerank").mock(
        return_value=httpx.Response(200, json=_sorted_payload([(2, 0.9), (0, 0.7)]))
    )
    from synapsekit.retrieval.voyage_reranker import VoyageReranker

    reranker = VoyageReranker(retriever=retriever, api_key="test-key")
    out = await reranker.retrieve("query", top_k=2)
    assert out == ["doc3", "doc1"]


@pytest.mark.asyncio
@respx.mock
async def test_voyage_retrieve_with_scores_async():
    retriever = _mock_retriever(["doc1", "doc2"])
    respx.post("https://api.voyageai.com/v1/rerank").mock(
        return_value=httpx.Response(200, json=_sorted_payload([(1, 0.8)]))
    )
    from synapsekit.retrieval.voyage_reranker import VoyageReranker

    reranker = VoyageReranker(retriever=retriever, api_key="test-key")
    out = await reranker.retrieve_with_scores("query", top_k=1)
    assert out == [{"text": "doc2", "relevance_score": 0.8}]


@pytest.mark.asyncio
@respx.mock
async def test_voyage_empty_candidates_short_circuits():
    retriever = _mock_retriever([])
    from synapsekit.retrieval.voyage_reranker import VoyageReranker

    reranker = VoyageReranker(retriever=retriever, api_key="test-key")
    assert await reranker.retrieve("query") == []
    assert await reranker.retrieve_with_scores("query") == []


def test_voyage_missing_key_raises():
    from synapsekit.retrieval.voyage_reranker import VoyageReranker

    reranker = VoyageReranker(retriever=_mock_retriever(["a"]))
    with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
        reranker._get_key()


# --------------------------------------------------------------------------- #
# JinaReranker
# --------------------------------------------------------------------------- #


@respx.mock
def test_jina_reranker_request_and_reorder():
    retriever = _mock_retriever(["doc1", "doc2", "doc3"])
    respx.post("https://api.jina.ai/v1/rerank").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"index": 2, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.7},
                ]
            },
        )
    )
    from synapsekit.retrieval.jina_reranker import JinaReranker

    reranker = JinaReranker(retriever=retriever, api_key="test-key")
    results = reranker._call("query", ["doc1", "doc2", "doc3"], top_n=2)
    assert results == [
        {"text": "doc3", "relevance_score": 0.9},
        {"text": "doc1", "relevance_score": 0.7},
    ]


@pytest.mark.asyncio
@respx.mock
async def test_jina_retrieve_async():
    retriever = _mock_retriever(["doc1", "doc2"])
    respx.post("https://api.jina.ai/v1/rerank").mock(
        return_value=httpx.Response(200, json={"results": [{"index": 1, "relevance_score": 0.8}]})
    )
    from synapsekit.retrieval.jina_reranker import JinaReranker

    reranker = JinaReranker(retriever=retriever, api_key="test-key")
    assert await reranker.retrieve("query", top_k=1) == ["doc2"]


def test_jina_missing_key_raises():
    from synapsekit.retrieval.jina_reranker import JinaReranker

    reranker = JinaReranker(retriever=_mock_retriever(["a"]))
    with pytest.raises(ValueError, match="JINA_API_KEY"):
        reranker._get_key()


# --------------------------------------------------------------------------- #
# MixedbreadReranker
# --------------------------------------------------------------------------- #


@respx.mock
def test_mixedbread_reranker_request_and_reorder():
    retriever = _mock_retriever(["doc1", "doc2", "doc3"])
    respx.post("https://api.mixedbread.ai/v1/rerank").mock(
        return_value=httpx.Response(200, json=_sorted_payload([(2, 0.9), (0, 0.7)]))
    )
    from synapsekit.retrieval.mixedbread_reranker import MixedbreadReranker

    reranker = MixedbreadReranker(retriever=retriever, api_key="test-key")
    results = reranker._call("query", ["doc1", "doc2", "doc3"], top_n=2)
    assert results == [
        {"text": "doc3", "relevance_score": 0.9},
        {"text": "doc1", "relevance_score": 0.7},
    ]


@pytest.mark.asyncio
@respx.mock
async def test_mixedbread_retrieve_async():
    retriever = _mock_retriever(["doc1", "doc2"])
    respx.post("https://api.mixedbread.ai/v1/rerank").mock(
        return_value=httpx.Response(200, json=_sorted_payload([(1, 0.8)]))
    )
    from synapsekit.retrieval.mixedbread_reranker import MixedbreadReranker

    reranker = MixedbreadReranker(retriever=retriever, api_key="test-key")
    assert await reranker.retrieve("query", top_k=1) == ["doc2"]


def test_mixedbread_missing_key_raises():
    from synapsekit.retrieval.mixedbread_reranker import MixedbreadReranker

    reranker = MixedbreadReranker(retriever=_mock_retriever(["a"]))
    with pytest.raises(ValueError, match="MXBAI_API_KEY"):
        reranker._get_key()
