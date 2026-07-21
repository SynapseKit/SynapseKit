"""Real ElasticsearchVectorStore integration tests (testcontainers).

Boots an actual Elasticsearch and exercises the real client path: true
nearest-neighbour kNN ranking, metadata filtering, top_k, persistence across a
reconnect, and empty/negative cases. Replaces MagicMock coverage. Part of #829.
"""

from __future__ import annotations

import inspect
import time

import numpy as np
import pytest

pytest.importorskip("elasticsearch")
_container_mod = pytest.importorskip("testcontainers.core.container")

from synapsekit.retrieval.elasticsearch_vector import ElasticsearchVectorStore  # noqa: E402

_ES_IMAGE = "docker.elastic.co/elasticsearch/elasticsearch:8.15.3"


class KeywordEmbeddings:
    """Deterministic real embeddings: one dimension per vocabulary word, so
    cosine similarity is exact word overlap and ranking is assertable."""

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
def es_url():
    from elasticsearch import Elasticsearch

    container = (
        _container_mod.DockerContainer(_ES_IMAGE)
        .with_env("discovery.type", "single-node")
        .with_env("xpack.security.enabled", "false")
        .with_env("ES_JAVA_OPTS", "-Xms512m -Xmx512m")
        .with_exposed_ports(9200)
    )
    with container as es:
        host = es.get_container_host_ip()
        port = es.get_exposed_port(9200)
        url = f"http://{host}:{port}"
        last_err: Exception | None = None
        for _ in range(120):
            try:
                client = Elasticsearch(hosts=[url])
                if client.ping():
                    client.close()
                    break
                client.close()
            except Exception as exc:
                last_err = exc
            time.sleep(1)
        else:
            raise RuntimeError(f"Elasticsearch did not become ready: {last_err}")
        yield url


def _store(url: str, index: str) -> ElasticsearchVectorStore:
    return ElasticsearchVectorStore(
        embedding_backend=KeywordEmbeddings(_VOCAB), url=url, index_name=index
    )


@pytest.mark.asyncio
async def test_add_and_search_returns_true_nearest_neighbor(es_url):
    store = _store(es_url, "rank_idx")
    await store.add(
        ["apple fruit red", "banana fruit yellow", "car vehicle fast"],
        [{"kind": "fruit"}, {"kind": "fruit"}, {"kind": "auto"}],
    )
    results = await store.search("apple", top_k=3)

    assert results, "expected non-empty results from a real Elasticsearch kNN query"
    assert results[0]["text"] == "apple fruit red"
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0]["metadata"] == {"kind": "fruit"}


@pytest.mark.asyncio
async def test_metadata_filter(es_url):
    store = _store(es_url, "filter_idx")
    await store.add(["apple fruit", "car vehicle"], [{"kind": "fruit"}, {"kind": "auto"}])
    results = await store.search("apple car", top_k=5, metadata_filter={"kind": "auto"})
    assert [r["text"] for r in results] == ["car vehicle"]


@pytest.mark.asyncio
async def test_top_k_limits_results(es_url):
    store = _store(es_url, "topk_idx")
    await store.add([f"apple fruit {w}" for w in ("red", "yellow", "cherry", "banana")])
    results = await store.search("apple", top_k=2)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_search_before_add_returns_empty(es_url):
    store = _store(es_url, "empty_idx")
    assert await store.search("apple") == []


@pytest.mark.asyncio
async def test_persistence_across_reconnect(es_url):
    writer = _store(es_url, "persist_idx")
    await writer.add(["apple fruit red"], [{"kind": "fruit"}])

    reader = _store(es_url, "persist_idx")
    results = await reader.search("apple", top_k=1)
    assert results[0]["text"] == "apple fruit red"


@pytest.mark.asyncio
async def test_add_empty_is_noop(es_url):
    store = _store(es_url, "noop_idx")
    await store.add([])


@pytest.mark.asyncio
async def test_metadata_length_mismatch_raises(es_url):
    store = _store(es_url, "mismatch_idx")
    with pytest.raises(ValueError):
        await store.add(["a", "b"], [{"only": "one"}])


def test_async_api_is_coroutine():
    # SynapseKit is async-first: the public IO surface must stay coroutines.
    store = ElasticsearchVectorStore(
        embedding_backend=KeywordEmbeddings(_VOCAB), url="http://localhost:9200"
    )
    assert inspect.iscoroutinefunction(store.add)
    assert inspect.iscoroutinefunction(store.search)
