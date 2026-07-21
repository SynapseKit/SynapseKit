"""Real ClickHouseVectorStore integration tests (testcontainers).

Boots a real ClickHouse and exercises the real client path against L2Distance
vector search: true nearest-neighbour ranking, metadata filtering, top_k,
reconnect persistence, empty/negative cases, and the async-first contract.
Part of #829.
"""

from __future__ import annotations

import inspect
import time

import numpy as np
import pytest

pytest.importorskip("clickhouse_connect")
_container_mod = pytest.importorskip("testcontainers.core.container")

from synapsekit.retrieval.clickhouse_vector import ClickHouseVectorStore  # noqa: E402

_CLICKHOUSE_IMAGE = "clickhouse/clickhouse-server:24.3"
_PASSWORD = "synapsekit"
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
def clickhouse_conn():
    import clickhouse_connect

    container = (
        _container_mod.DockerContainer(_CLICKHOUSE_IMAGE)
        .with_env("CLICKHOUSE_USER", "default")
        .with_env("CLICKHOUSE_PASSWORD", _PASSWORD)
        .with_env("CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT", "1")
        .with_exposed_ports(8123)
    )
    with container as c:
        host = c.get_container_host_ip()
        port = int(c.get_exposed_port(8123))
        last_err: Exception | None = None
        for _ in range(60):
            try:
                client = clickhouse_connect.get_client(host=host, port=port, password=_PASSWORD)
                client.command("SELECT 1")
                client.close()
                break
            except Exception as exc:
                last_err = exc
                time.sleep(1)
        else:
            raise RuntimeError(f"clickhouse did not become ready: {last_err}")
        yield {"host": host, "port": port}


def _store(conn: dict, table: str) -> ClickHouseVectorStore:
    return ClickHouseVectorStore(
        embedding_backend=KeywordEmbeddings(_VOCAB),
        host=conn["host"],
        port=conn["port"],
        table_name=table,
        password=_PASSWORD,
    )


@pytest.mark.asyncio
async def test_add_and_search_returns_true_nearest_neighbor(clickhouse_conn):
    store = _store(clickhouse_conn, "rank")
    await store.add(
        ["apple fruit red", "banana fruit yellow", "car vehicle fast"],
        [{"kind": "fruit"}, {"kind": "fruit"}, {"kind": "auto"}],
    )
    results = await store.search("apple", top_k=3)
    assert results, "expected non-empty results from a real ClickHouse query"
    assert results[0]["text"] == "apple fruit red"
    scores = [r["score"] for r in results]
    assert scores == sorted(scores)  # L2 distance -> ascending (nearest first)
    assert results[0]["metadata"] == {"kind": "fruit"}


@pytest.mark.asyncio
async def test_metadata_filter(clickhouse_conn):
    store = _store(clickhouse_conn, "filter_t")
    await store.add(["apple fruit", "car vehicle"], [{"kind": "fruit"}, {"kind": "auto"}])
    results = await store.search("apple car", top_k=5, metadata_filter={"kind": "auto"})
    assert [r["text"] for r in results] == ["car vehicle"]


@pytest.mark.asyncio
async def test_top_k_limits_results(clickhouse_conn):
    store = _store(clickhouse_conn, "topk")
    await store.add([f"apple fruit {w}" for w in ("red", "yellow", "cherry", "banana")])
    results = await store.search("apple", top_k=2)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_search_before_add_returns_empty(clickhouse_conn):
    store = _store(clickhouse_conn, "empty_first")
    assert await store.search("apple") == []


@pytest.mark.asyncio
async def test_persistence_across_reconnect(clickhouse_conn):
    writer = _store(clickhouse_conn, "persist")
    await writer.add(["apple fruit red"], [{"kind": "fruit"}])

    reader = _store(clickhouse_conn, "persist")
    results = await reader.search("apple", top_k=1)
    assert results[0]["text"] == "apple fruit red"


@pytest.mark.asyncio
async def test_add_empty_is_noop(clickhouse_conn):
    store = _store(clickhouse_conn, "noop")
    await store.add([])


@pytest.mark.asyncio
async def test_top_k_injection_rejected(clickhouse_conn):
    store = _store(clickhouse_conn, "inject")
    await store.add(["apple fruit red"], [{"kind": "fruit"}])
    with pytest.raises((ValueError, TypeError)):
        await store.search("apple", top_k="5; DROP TABLE inject")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_metadata_length_mismatch_raises(clickhouse_conn):
    store = _store(clickhouse_conn, "mismatch")
    with pytest.raises(ValueError):
        await store.add(["a", "b"], [{"only": "one"}])


def test_async_api_is_coroutine():
    # SynapseKit is async-first: the public IO surface must stay coroutines.
    assert inspect.iscoroutinefunction(ClickHouseVectorStore.add)
    assert inspect.iscoroutinefunction(ClickHouseVectorStore.search)
