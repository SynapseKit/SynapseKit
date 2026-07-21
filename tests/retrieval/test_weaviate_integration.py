"""Real WeaviateVectorStore integration tests (testcontainers, weaviate-client v4).

Boots a real Weaviate (http + gRPC) and exercises the real client path: true
nearest-neighbour ranking, native metadata filtering, top_k, reconnect
persistence, empty/negative cases, and the async-first contract. Regression
coverage for #838. Part of #829.
"""

from __future__ import annotations

import contextlib
import inspect
import time

import numpy as np
import pytest

pytest.importorskip("weaviate")
_container_mod = pytest.importorskip("testcontainers.core.container")

from synapsekit.retrieval.weaviate import WeaviateVectorStore  # noqa: E402

_WEAVIATE_IMAGE = "cr.weaviate.io/semitechnologies/weaviate:1.27.0"
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


def _connect(conn: dict):
    import weaviate

    return weaviate.connect_to_custom(
        http_host=conn["host"],
        http_port=conn["http"],
        http_secure=False,
        grpc_host=conn["host"],
        grpc_port=conn["grpc"],
        grpc_secure=False,
    )


@pytest.fixture(scope="module")
def weaviate_conn():
    container = (
        _container_mod.DockerContainer(_WEAVIATE_IMAGE)
        .with_env("AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED", "true")
        .with_env("PERSISTENCE_DATA_PATH", "/var/lib/weaviate")
        .with_env("DEFAULT_VECTORIZER_MODULE", "none")
        .with_env("QUERY_DEFAULTS_LIMIT", "25")
        .with_exposed_ports(8080, 50051)
    )
    with container as c:
        conn = {
            "host": c.get_container_host_ip(),
            "http": int(c.get_exposed_port(8080)),
            "grpc": int(c.get_exposed_port(50051)),
        }
        client = None
        for _ in range(60):
            try:
                client = _connect(conn)
                if client.is_ready():
                    break
                client.close()
            except Exception:
                if client is not None:
                    with contextlib.suppress(Exception):
                        client.close()
                time.sleep(1)
        else:
            raise RuntimeError("weaviate did not become ready")
        try:
            conn["client"] = client
            yield conn
        finally:
            client.close()


def _store(conn: dict, collection: str, client=None) -> WeaviateVectorStore:
    return WeaviateVectorStore(
        embedding_backend=KeywordEmbeddings(_VOCAB),
        collection_name=collection,
        client=client or conn["client"],
    )


@pytest.mark.asyncio
async def test_add_and_search_returns_true_nearest_neighbor(weaviate_conn):
    store = _store(weaviate_conn, "Rank")
    await store.add(
        ["apple fruit red", "banana fruit yellow", "car vehicle fast"],
        [{"kind": "fruit"}, {"kind": "fruit"}, {"kind": "auto"}],
    )
    results = await store.search("apple", top_k=3)
    assert results, "expected non-empty results from a real Weaviate query"
    assert results[0]["text"] == "apple fruit red"
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0]["metadata"] == {"kind": "fruit"}


@pytest.mark.asyncio
async def test_metadata_filter(weaviate_conn):
    store = _store(weaviate_conn, "FilterC")
    await store.add(["apple fruit", "car vehicle"], [{"kind": "fruit"}, {"kind": "auto"}])
    results = await store.search("apple car", top_k=5, metadata_filter={"kind": "auto"})
    assert [r["text"] for r in results] == ["car vehicle"]


@pytest.mark.asyncio
async def test_top_k_limits_results(weaviate_conn):
    store = _store(weaviate_conn, "Topk")
    await store.add([f"apple fruit {w}" for w in ("red", "yellow", "cherry", "banana")])
    results = await store.search("apple", top_k=2)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_search_before_add_returns_empty(weaviate_conn):
    store = _store(weaviate_conn, "EmptyFirst")
    assert await store.search("apple") == []


@pytest.mark.asyncio
async def test_persistence_across_reconnect(weaviate_conn):
    writer = _store(weaviate_conn, "Persist")
    await writer.add(["apple fruit red"], [{"kind": "fruit"}])

    # Fresh client (new gRPC connection) must see the persisted object.
    reader_client = _connect(weaviate_conn)
    try:
        reader = _store(weaviate_conn, "Persist", client=reader_client)
        results = await reader.search("apple", top_k=1)
        assert results[0]["text"] == "apple fruit red"
    finally:
        reader_client.close()


@pytest.mark.asyncio
async def test_add_empty_is_noop(weaviate_conn):
    store = _store(weaviate_conn, "Noop")
    await store.add([])


@pytest.mark.asyncio
async def test_metadata_length_mismatch_raises(weaviate_conn):
    store = _store(weaviate_conn, "Mismatch")
    with pytest.raises(ValueError):
        await store.add(["a", "b"], [{"only": "one"}])


def test_async_api_is_coroutine():
    # SynapseKit is async-first: the public IO surface must stay coroutines.
    class _FakeClient:
        pass

    store = WeaviateVectorStore(embedding_backend=KeywordEmbeddings(_VOCAB), client=_FakeClient())
    assert inspect.iscoroutinefunction(store.add)
    assert inspect.iscoroutinefunction(store.search)
