"""Auto-instrumentation of the embeddings/reranker provider layer (#886) → Live.

Real subclasses only — a hand-written ``BaseEmbeddings`` and a real ``Reranker``
over a tiny in-memory retriever. No server, no mocks: instrument the classes and
toggle the bus directly, mirroring ``test_instrument.py``.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from synapsekit.embeddings.base import BaseEmbeddings
from synapsekit.live import bus
from synapsekit.live.instrument import instrument_all
from synapsekit.retrieval.reranker import Reranker


class _FakeEmbeddings(BaseEmbeddings):
    async def _embed_raw(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 3), dtype=np.float32)


class _FakeRetriever:
    async def retrieve(self, query: str, top_k: int = 5, metadata_filter=None) -> list[str]:
        return ["doc a", "doc b"][:top_k]


class _FakeReranker(Reranker):
    async def retrieve(self, query: str, top_k: int = 5, metadata_filter=None) -> list[str]:
        return await self._retriever.retrieve(query, top_k=top_k)

    async def retrieve_with_scores(
        self, query: str, top_k: int = 5, metadata_filter=None
    ) -> list[dict]:
        docs = await self._retriever.retrieve(query, top_k=top_k)
        return [{"text": doc, "relevance_score": 1.0} for doc in docs]


@pytest.fixture(autouse=True)
def _instrumented_bus():
    instrument_all()  # idempotent — patches the base classes + future subclasses
    was = bus.enabled
    bus.enabled = True
    bus.clear()
    yield
    bus.enabled = was


def test_embed_stays_a_coroutine_after_instrumentation() -> None:
    assert inspect.iscoroutinefunction(BaseEmbeddings.embed)
    assert inspect.iscoroutinefunction(_FakeEmbeddings.embed)


async def test_embed_publishes_event() -> None:
    await _FakeEmbeddings().embed(["a", "b"])
    events = [e for e in bus.history() if e["kind"] == "embeddings.embed"]
    assert events, "embeddings.embed not published"
    assert events[-1]["attributes"]["provider"] == "_FakeEmbeddings"
    assert events[-1]["attributes"]["count"] == 2
    assert events[-1]["status"] == "ok"
    assert events[-1]["duration_ms"] >= 0


async def test_reranker_publishes_event() -> None:
    reranker = _FakeReranker(_FakeRetriever(), model="fake-rerank")  # type: ignore[arg-type]
    await reranker.retrieve("q", top_k=2)
    await reranker.retrieve_with_scores("q", top_k=2)
    events = [e for e in bus.history() if e["kind"] == "rerank"]
    assert len(events) >= 2, "rerank not published for both entry points"
    assert events[-1]["attributes"]["reranker"] == "_FakeReranker"


async def test_embed_error_is_reported_and_reraised() -> None:
    class _BoomEmbeddings(BaseEmbeddings):
        async def _embed_raw(self, texts: list[str]) -> np.ndarray:
            raise RuntimeError("kaboom")

    with pytest.raises(RuntimeError, match="kaboom"):
        await _BoomEmbeddings().embed(["x"])
    events = [e for e in bus.history() if e["kind"] == "embeddings.embed"]
    assert events[-1]["status"] == "error"
