"""Real QdrantVectorStore integration tests (testcontainers).

Boots an actual Qdrant and exercises the real client path: true nearest-neighbour
ranking, metadata filtering, top_k, append-on-reconnect persistence, empty/negative
cases, and the async-first contract. Regression coverage for #836. Part of #829.
"""

from __future__ import annotations

import inspect
import time

import numpy as np
import pytest

pytest.importorskip("qdrant_client")
_container_mod = pytest.importorskip("testcontainers.core.container")

from synapsekit.retrieval.qdrant import QdrantVectorStore  # noqa: E402

_QDRANT_IMAGE = "qdrant/qdrant:latest"
_VOCAB = ["apple", "banana", "cherry", "fruit", "red", "yellow", "car", "vehicle", "fast"]


class KeywordEmbeddings:
    def __init__(self, vocab: list[str]) -> None:
        self._index = {word: i for i, word in enumerate(vocab)}
        self.dimension = len(vocab)

    def _one(self, text: str) -> np.ndarray:
        vec = np.zeros(len(self._index), dtype=np.float32)
        for token in text.lower().split():
            if token in self._index:
                vec[self._index[token]] += 1.0
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, len(self._index)), dtype=np.float32)
        return np.vstack([self._one(t) for t in texts]).astype(np.float32)

    async def embed_one(self, text: str) -> np.ndarray:
        return self._one(text)


@pytest.fixture(scope="module")
def qdrant_url():
    from qdrant_client import QdrantClient

    container = _container_mod.DockerContainer(_QDRANT_IMAGE).with_exposed_ports(6333)
    with container as c:
        host = c.get_container_host_ip()
        port = c.get_exposed_port(6333)
        url = f"http://{host}:{port}"
        last_err: Exception | None = None
        for _ in range(60):
            try:
                client = QdrantClient(url=url)
                client.get_collections()
                client.close()
                break
            except Exception as exc:
                last_err = exc
                time.sleep(1)
        else:
            raise RuntimeError(f"qdrant did not become ready: {last_err}")
        yield url


def _store(url: str, collection: str) -> QdrantVectorStore:
    return QdrantVectorStore(
        embedding_backend=KeywordEmbeddings(_VOCAB), url=url, collection_name=collection
    )


@pytest.mark.asyncio
async def test_add_and_search_returns_true_nearest_neighbor(qdrant_url):
    store = _store(qdrant_url, "rank")
    await store.add(
        ["apple fruit red", "banana fruit yellow", "car vehicle fast"],
        [{"kind": "fruit"}, {"kind": "fruit"}, {"kind": "auto"}],
    )
    results = await store.search("apple", top_k=3)
    assert results, "expected non-empty results from a real Qdrant query"
    assert results[0]["text"] == "apple fruit red"
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)  # cosine similarity -> descending
    assert results[0]["metadata"] == {"kind": "fruit"}


@pytest.mark.asyncio
async def test_metadata_filter(qdrant_url):
    store = _store(qdrant_url, "filter")
    await store.add(["apple fruit", "car vehicle"], [{"kind": "fruit"}, {"kind": "auto"}])
    results = await store.search("apple car", top_k=5, metadata_filter={"kind": "auto"})
    assert [r["text"] for r in results] == ["car vehicle"]


@pytest.mark.asyncio
async def test_top_k_limits_results(qdrant_url):
    store = _store(qdrant_url, "topk")
    await store.add([f"apple fruit {w}" for w in ("red", "yellow", "cherry", "banana")])
    results = await store.search("apple", top_k=2)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_search_before_add_returns_empty(qdrant_url):
    store = _store(qdrant_url, "empty_first")
    assert await store.search("apple") == []


@pytest.mark.asyncio
async def test_reconnect_appends_not_overwrites(qdrant_url):
    # Regression for the id-collision bug: a fresh instance must append, keeping
    # the first writer's doc searchable rather than overwriting id 0,1,2...
    writer1 = _store(qdrant_url, "persist")
    await writer1.add(["apple fruit red"], [{"kind": "fruit"}])
    writer2 = _store(qdrant_url, "persist")
    await writer2.add(["apple fruit cherry"], [{"kind": "fruit"}])

    reader = _store(qdrant_url, "persist")
    results = await reader.search("apple", top_k=5)
    texts = {r["text"] for r in results}
    assert texts == {"apple fruit red", "apple fruit cherry"}  # both survive


@pytest.mark.asyncio
async def test_add_empty_is_noop(qdrant_url):
    store = _store(qdrant_url, "noop")
    await store.add([])


@pytest.mark.asyncio
async def test_metadata_length_mismatch_raises(qdrant_url):
    store = _store(qdrant_url, "mismatch")
    with pytest.raises(ValueError):
        await store.add(["a", "b"], [{"only": "one"}])


def test_async_api_is_coroutine():
    # SynapseKit is async-first: the public IO surface must stay coroutines.
    store = QdrantVectorStore(
        embedding_backend=KeywordEmbeddings(_VOCAB), url="http://localhost:6333"
    )
    assert inspect.iscoroutinefunction(store.add)
    assert inspect.iscoroutinefunction(store.search)
