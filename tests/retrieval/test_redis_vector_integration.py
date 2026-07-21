"""Real RedisVectorStore integration tests (testcontainers), replacing MagicMock.

Boots an actual ``redis/redis-stack`` (RediSearch) and exercises the real client
path: true nearest-neighbour ranking (cosine distance, ascending), metadata
filtering, top_k, persistence across a reconnect, and empty/negative cases.
Skips cleanly when Docker or the deps are unavailable. Part of epic #829.
"""

from __future__ import annotations

import inspect
import time

import numpy as np
import pytest

pytest.importorskip("redis")
_container_mod = pytest.importorskip("testcontainers.core.container")

from synapsekit.retrieval.redis_vector import RedisVectorStore  # noqa: E402

_REDIS_IMAGE = "redis/redis-stack:latest"


class KeywordEmbeddings:
    """Deterministic real embeddings: one dimension per vocabulary word, so
    cosine distance is exact word overlap and ranking is assertable."""

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


_VOCAB = ["apple", "banana", "cherry", "fruit", "red", "yellow", "car", "vehicle", "fast"]


@pytest.fixture(scope="module")
def redis_url():
    import redis as redis_mod

    container = _container_mod.DockerContainer(_REDIS_IMAGE).with_exposed_ports(6379)
    with container as c:
        host = c.get_container_host_ip()
        port = c.get_exposed_port(6379)
        url = f"redis://{host}:{port}"
        last_err: Exception | None = None
        for _ in range(60):
            try:
                client = redis_mod.from_url(url)
                client.execute_command("FT._LIST")  # RediSearch module loaded?
                client.close()
                break
            except Exception as exc:
                last_err = exc
                time.sleep(1)
        else:
            raise RuntimeError(f"redis-stack did not become ready: {last_err}")
        yield url


def _store(url: str, index: str) -> RedisVectorStore:
    return RedisVectorStore(embedding_backend=KeywordEmbeddings(_VOCAB), url=url, index_name=index)


@pytest.mark.asyncio
async def test_add_and_search_returns_true_nearest_neighbor(redis_url):
    store = _store(redis_url, "rank_idx")
    await store.add(
        ["apple fruit red", "banana fruit yellow", "car vehicle fast"],
        [{"kind": "fruit"}, {"kind": "fruit"}, {"kind": "auto"}],
    )
    results = await store.search("apple", top_k=3)

    assert results, "expected non-empty results from a real RediSearch query"
    assert results[0]["text"] == "apple fruit red"  # exact nearest neighbour
    scores = [r["score"] for r in results]
    assert scores == sorted(scores)  # cosine distance -> ascending (nearest first)
    assert results[0]["metadata"] == {"kind": "fruit"}


@pytest.mark.asyncio
async def test_metadata_filter(redis_url):
    store = _store(redis_url, "filter_idx")
    await store.add(["apple fruit", "car vehicle"], [{"kind": "fruit"}, {"kind": "auto"}])
    results = await store.search("apple car", top_k=5, metadata_filter={"kind": "auto"})
    assert [r["text"] for r in results] == ["car vehicle"]


@pytest.mark.asyncio
async def test_top_k_limits_results(redis_url):
    store = _store(redis_url, "topk_idx")
    await store.add([f"apple fruit {w}" for w in ("red", "yellow", "cherry", "banana")])
    results = await store.search("apple", top_k=2)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_search_before_add_returns_empty(redis_url):
    store = _store(redis_url, "empty_idx")
    assert await store.search("apple") == []


@pytest.mark.asyncio
async def test_persistence_across_reconnect(redis_url):
    writer = _store(redis_url, "persist_idx")
    await writer.add(["apple fruit red"], [{"kind": "fruit"}])

    # Fresh instance (new client, no prior add) must find the persisted vector.
    reader = _store(redis_url, "persist_idx")
    results = await reader.search("apple", top_k=1)
    assert results[0]["text"] == "apple fruit red"


@pytest.mark.asyncio
async def test_add_empty_is_noop(redis_url):
    store = _store(redis_url, "noop_idx")
    await store.add([])


@pytest.mark.asyncio
async def test_metadata_length_mismatch_raises(redis_url):
    store = _store(redis_url, "mismatch_idx")
    with pytest.raises(ValueError):
        await store.add(["a", "b"], [{"only": "one"}])


def test_async_api_is_coroutine():
    # SynapseKit is async-first: the public IO surface must stay coroutines.
    store = RedisVectorStore(
        embedding_backend=KeywordEmbeddings(_VOCAB), url="redis://localhost:6379"
    )
    assert inspect.iscoroutinefunction(store.add)
    assert inspect.iscoroutinefunction(store.search)
