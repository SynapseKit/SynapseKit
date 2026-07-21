"""Real OpenSearchVectorStore integration tests (testcontainers).

Boots a real OpenSearch and exercises the real client path against the k-NN
plugin: true nearest-neighbour ranking, metadata filtering, top_k, reconnect
persistence, empty/negative cases, and the async-first contract. Part of #829.
"""

from __future__ import annotations

import inspect
import time

import numpy as np
import pytest

pytest.importorskip("opensearchpy")
_container_mod = pytest.importorskip("testcontainers.core.container")

from synapsekit.retrieval.opensearch_vector import OpenSearchVectorStore  # noqa: E402

_OPENSEARCH_IMAGE = "opensearchproject/opensearch:2.17.0"
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
def opensearch_url():
    from opensearchpy import OpenSearch

    container = (
        _container_mod.DockerContainer(_OPENSEARCH_IMAGE)
        .with_env("discovery.type", "single-node")
        .with_env("DISABLE_SECURITY_PLUGIN", "true")
        .with_env("DISABLE_INSTALL_DEMO_CONFIG", "true")
        .with_env("OPENSEARCH_JAVA_OPTS", "-Xms512m -Xmx512m")
        .with_exposed_ports(9200)
    )
    with container as c:
        host = c.get_container_host_ip()
        port = int(c.get_exposed_port(9200))
        url = f"http://{host}:{port}"
        last_err: Exception | None = None
        for _ in range(120):
            try:
                client = OpenSearch(hosts=[url], use_ssl=False, verify_certs=False)
                if client.ping():
                    break
            except Exception as exc:
                last_err = exc
            time.sleep(1)
        else:
            raise RuntimeError(f"opensearch did not become ready: {last_err}")
        yield url


def _store(url: str, index: str) -> OpenSearchVectorStore:
    return OpenSearchVectorStore(
        embedding_backend=KeywordEmbeddings(_VOCAB), url=url, index_name=index
    )


@pytest.mark.asyncio
async def test_add_and_search_returns_true_nearest_neighbor(opensearch_url):
    store = _store(opensearch_url, "rank")
    await store.add(
        ["apple fruit red", "banana fruit yellow", "car vehicle fast"],
        [{"kind": "fruit"}, {"kind": "fruit"}, {"kind": "auto"}],
    )
    results = await store.search("apple", top_k=3)
    assert results, "expected non-empty results from a real OpenSearch k-NN query"
    assert results[0]["text"] == "apple fruit red"
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0]["metadata"] == {"kind": "fruit"}


@pytest.mark.asyncio
async def test_metadata_filter(opensearch_url):
    store = _store(opensearch_url, "filter_t")
    await store.add(["apple fruit", "car vehicle"], [{"kind": "fruit"}, {"kind": "auto"}])
    results = await store.search("apple car", top_k=5, metadata_filter={"kind": "auto"})
    assert [r["text"] for r in results] == ["car vehicle"]


@pytest.mark.asyncio
async def test_top_k_limits_results(opensearch_url):
    store = _store(opensearch_url, "topk")
    await store.add([f"apple fruit {w}" for w in ("red", "yellow", "cherry", "banana")])
    results = await store.search("apple", top_k=2)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_search_before_add_returns_empty(opensearch_url):
    store = _store(opensearch_url, "empty_first")
    assert await store.search("apple") == []


@pytest.mark.asyncio
async def test_persistence_across_reconnect(opensearch_url):
    writer = _store(opensearch_url, "persist")
    await writer.add(["apple fruit red"], [{"kind": "fruit"}])

    reader = _store(opensearch_url, "persist")
    results = await reader.search("apple", top_k=1)
    assert results[0]["text"] == "apple fruit red"


@pytest.mark.asyncio
async def test_add_empty_is_noop(opensearch_url):
    store = _store(opensearch_url, "noop")
    await store.add([])


@pytest.mark.asyncio
async def test_metadata_length_mismatch_raises(opensearch_url):
    store = _store(opensearch_url, "mismatch")
    with pytest.raises(ValueError):
        await store.add(["a", "b"], [{"only": "one"}])


def test_async_api_is_coroutine():
    # SynapseKit is async-first: the public IO surface must stay coroutines.
    assert inspect.iscoroutinefunction(OpenSearchVectorStore.add)
    assert inspect.iscoroutinefunction(OpenSearchVectorStore.search)
