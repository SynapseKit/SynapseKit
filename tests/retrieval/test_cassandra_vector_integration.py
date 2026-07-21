"""Real CassandraVectorStore integration tests (testcontainers, cassandra-driver mode).

Boots a real Cassandra 5.0 and exercises the real driver path against native ANN
vector search (`vector<float, n>` + StorageAttachedIndex): true nearest-neighbour
ranking, metadata filtering, top_k, reconnect persistence, empty/negative cases,
and the async-first contract. Regression coverage for #842. Part of #829.
"""

from __future__ import annotations

import inspect
import time

import numpy as np
import pytest

pytest.importorskip("cassandra")
_container_mod = pytest.importorskip("testcontainers.core.container")

from synapsekit.retrieval.cassandra_vector import CassandraVectorStore  # noqa: E402

_CASSANDRA_IMAGE = "cassandra:5"
_KEYSPACE = "synapsekit_test"
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


def _make_cluster(host: str, port: int):
    from cassandra.cluster import Cluster
    from cassandra.policies import AddressTranslator

    class _StaticTranslator(AddressTranslator):
        # testcontainers maps 9042 to a random host port; the node advertises its
        # container-internal address, so translate every discovered address back
        # to the reachable host (the mapped port is the Cluster's `port`).
        def translate(self, addr):
            return host

    return Cluster(
        contact_points=[host],
        port=port,
        address_translator=_StaticTranslator(),
    )


@pytest.fixture(scope="module")
def cassandra_session():
    container = _container_mod.DockerContainer(_CASSANDRA_IMAGE).with_exposed_ports(9042)
    with container as c:
        host = c.get_container_host_ip()
        port = int(c.get_exposed_port(9042))
        cluster = None
        session = None
        for _ in range(120):
            try:
                cluster = _make_cluster(host, port)
                session = cluster.connect()
                session.execute(
                    f"CREATE KEYSPACE IF NOT EXISTS {_KEYSPACE} "
                    "WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1}"
                )
                break
            except Exception:
                if cluster is not None:
                    cluster.shutdown()
                time.sleep(2)
        else:
            raise RuntimeError("cassandra did not become ready")
        try:
            yield {"host": host, "port": port, "session": session}
        finally:
            cluster.shutdown()


def _store(conn: dict, table: str, session=None) -> CassandraVectorStore:
    return CassandraVectorStore(
        embedding_backend=KeywordEmbeddings(_VOCAB),
        keyspace=_KEYSPACE,
        table_name=table,
        session=session or conn["session"],
    )


async def _search_until(store, query, expected_text, *, top_k=5, metadata_filter=None):
    """Poll until the SAI index has indexed the new rows (build is async)."""
    for _ in range(30):
        results = await store.search(query, top_k=top_k, metadata_filter=metadata_filter)
        if any(r["text"] == expected_text for r in results):
            return results
        time.sleep(1)
    return await store.search(query, top_k=top_k, metadata_filter=metadata_filter)


@pytest.mark.asyncio
async def test_add_and_search_returns_true_nearest_neighbor(cassandra_session):
    store = _store(cassandra_session, "rank")
    await store.add(
        ["apple fruit red", "banana fruit yellow", "car vehicle fast"],
        [{"kind": "fruit"}, {"kind": "fruit"}, {"kind": "auto"}],
    )
    results = await _search_until(store, "apple", "apple fruit red", top_k=3)
    assert results[0]["text"] == "apple fruit red"
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)  # real cosine similarity, descending
    assert results[0]["metadata"] == {"kind": "fruit"}


@pytest.mark.asyncio
async def test_metadata_filter(cassandra_session):
    store = _store(cassandra_session, "filter_t")
    await store.add(["apple fruit", "car vehicle"], [{"kind": "fruit"}, {"kind": "auto"}])
    results = await _search_until(
        store, "apple car", "car vehicle", top_k=5, metadata_filter={"kind": "auto"}
    )
    assert [r["text"] for r in results] == ["car vehicle"]


@pytest.mark.asyncio
async def test_search_before_add_returns_empty(cassandra_session):
    store = _store(cassandra_session, "empty_first")
    assert await store.search("apple") == []


@pytest.mark.asyncio
async def test_persistence_across_reconnect(cassandra_session):
    writer = _store(cassandra_session, "persist")
    await writer.add(["apple fruit red"], [{"kind": "fruit"}])

    # Fresh session (new cluster connection) must see the persisted row.
    reader_cluster = _make_cluster(cassandra_session["host"], cassandra_session["port"])
    reader_session = reader_cluster.connect()
    try:
        reader = _store(cassandra_session, "persist", session=reader_session)
        results = await _search_until(reader, "apple", "apple fruit red", top_k=1)
        assert results[0]["text"] == "apple fruit red"
    finally:
        reader_cluster.shutdown()


@pytest.mark.asyncio
async def test_add_empty_is_noop(cassandra_session):
    store = _store(cassandra_session, "noop")
    await store.add([])


@pytest.mark.asyncio
async def test_metadata_length_mismatch_raises(cassandra_session):
    store = _store(cassandra_session, "mismatch")
    with pytest.raises(ValueError):
        await store.add(["a", "b"], [{"only": "one"}])


def test_async_api_is_coroutine():
    # SynapseKit is async-first: the public IO surface must stay coroutines.
    class _FakeSession:
        pass

    store = CassandraVectorStore(
        embedding_backend=KeywordEmbeddings(_VOCAB),
        keyspace=_KEYSPACE,
        session=_FakeSession(),
    )
    assert inspect.iscoroutinefunction(store.add)
    assert inspect.iscoroutinefunction(store.search)
