"""Tests for FlashRankReranker."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from synapsekit import FlashRankReranker
from synapsekit.retrieval.reranker import Reranker


class FakeRetriever:
    """Hand-written fake retriever for testing without external services."""

    def __init__(self, items: list[str | dict]) -> None:
        self.items = items

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[str]:
        if self.items and isinstance(self.items[0], dict):
            return [x["text"] for x in self.items][:top_k]
        return [str(x) for x in self.items][:top_k]

    async def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        if self.items and isinstance(self.items[0], dict):
            return list(self.items)[:top_k]
        return [{"text": str(x), "relevance_score": 0.5} for x in self.items][:top_k]


class FakeSimpleRetriever:
    """Retriever providing only `retrieve` (no `retrieve_with_scores`)."""

    def __init__(self, texts: list[str]) -> None:
        self.texts = texts

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[str]:
        return list(self.texts)[:top_k]


class FakeRerankRequest:
    def __init__(self, query: str, passages: list[dict]) -> None:
        self.query = query
        self.passages = passages


class FakeRanker:
    def __init__(
        self,
        model_name: str | None = None,
        cache_dir: str | None = None,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.max_length = max_length

    def rerank(self, request: FakeRerankRequest) -> list[dict]:
        scores: dict[str, float] = {
            "apple": 0.2,
            "banana": 0.95,
            "cherry": 0.65,
        }
        scored: list[dict] = []
        for p in request.passages:
            item = dict(p)
            text = str(item.get("text", ""))
            item["score"] = scores.get(text, 0.5)
            scored.append(item)
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored


@pytest.fixture
def fake_flashrank(monkeypatch) -> FakeRanker:
    fake_mod = ModuleType("flashrank")
    ranker_instance = FakeRanker()

    def ranker_factory(
        model_name: str | None = None,
        cache_dir: str | None = None,
        max_length: int = 512,
    ) -> FakeRanker:
        ranker_instance.model_name = model_name
        ranker_instance.cache_dir = cache_dir
        ranker_instance.max_length = max_length
        return ranker_instance

    fake_mod.Ranker = ranker_factory  # type: ignore[attr-defined]
    fake_mod.RerankRequest = FakeRerankRequest  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "flashrank", fake_mod)
    return ranker_instance


def test_flashrank_subclasses_reranker() -> None:
    assert issubclass(FlashRankReranker, Reranker)


def test_flashrank_missing_dependency_raises_clear_error(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "flashrank", None)
    reranker = FlashRankReranker(retriever=FakeRetriever(["test"]))
    with pytest.raises(ImportError, match="pip install synapsekit\\[flashrank\\]"):
        reranker._get_ranker()


@pytest.mark.asyncio
async def test_flashrank_empty_candidates_short_circuits() -> None:
    reranker = FlashRankReranker(retriever=FakeRetriever([]))
    assert await reranker.retrieve("What is RAG?") == []
    assert await reranker.retrieve_with_scores("What is RAG?") == []


@pytest.mark.asyncio
async def test_flashrank_retrieve_reorders_candidates(fake_flashrank) -> None:
    retriever = FakeRetriever(["apple", "banana", "cherry"])
    reranker = FlashRankReranker(retriever=retriever, fetch_k=10)

    results = await reranker.retrieve("fruits", top_k=2)
    assert results == ["banana", "cherry"]


@pytest.mark.asyncio
async def test_flashrank_retrieve_with_scores(fake_flashrank) -> None:
    retriever = FakeRetriever(["apple", "banana", "cherry"])
    reranker = FlashRankReranker(retriever=retriever, fetch_k=10)

    results = await reranker.retrieve_with_scores("fruits", top_k=2)
    assert len(results) == 2
    assert results[0]["text"] == "banana"
    assert results[0]["relevance_score"] == pytest.approx(0.95)
    assert results[1]["text"] == "cherry"
    assert results[1]["relevance_score"] == pytest.approx(0.65)


@pytest.mark.asyncio
async def test_flashrank_preserves_metadata(fake_flashrank) -> None:
    candidates = [
        {"text": "apple", "source": "orchard.txt", "doc_id": "1"},
        {"text": "banana", "source": "tropical.txt", "doc_id": "2"},
    ]
    retriever = FakeRetriever(candidates)
    reranker = FlashRankReranker(retriever=retriever)

    results = await reranker.retrieve_with_scores("query", top_k=2)
    assert results[0]["text"] == "banana"
    assert results[0]["source"] == "tropical.txt"
    assert results[0]["doc_id"] == "2"
    assert results[0]["relevance_score"] == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_flashrank_custom_init_parameters(fake_flashrank) -> None:
    retriever = FakeRetriever(["apple"])
    reranker = FlashRankReranker(
        retriever=retriever,
        model="custom-onnx-model",
        cache_dir="/tmp/models",
        max_length=256,
    )
    await reranker.retrieve("query")
    assert fake_flashrank.model_name == "custom-onnx-model"
    assert fake_flashrank.cache_dir == "/tmp/models"
    assert fake_flashrank.max_length == 256


@pytest.mark.asyncio
async def test_flashrank_fallback_when_retriever_lacks_retrieve_with_scores(
    fake_flashrank,
) -> None:
    retriever = FakeSimpleRetriever(["apple", "banana"])
    reranker = FlashRankReranker(retriever=retriever)

    scored = await reranker.retrieve_with_scores("query", top_k=1)
    assert len(scored) == 1
    assert scored[0]["text"] == "banana"
    assert scored[0]["relevance_score"] == pytest.approx(0.95)
