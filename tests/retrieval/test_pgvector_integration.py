"""Real pgvector integration tests (testcontainers), replacing MagicMock coverage.

Boots an actual ``pgvector/pgvector:pg16`` Postgres and exercises the real
psycopg path: true nearest-neighbour ranking, metadata filtering, top_k,
persistence across a reconnect, and empty/negative cases. Regression coverage
for #830 (dimension / register_vector / commit / opclass / JSONB bugs the mock
hid). Skips cleanly when Docker or the deps are unavailable.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

pytest.importorskip("psycopg")
pytest.importorskip("pgvector")
_container_mod = pytest.importorskip("testcontainers.core.container")

from synapsekit.retrieval.pgvector import DistanceStrategy, PGVectorStore  # noqa: E402

_PG_IMAGE = "pgvector/pgvector:pg16"


class KeywordEmbeddings:
    """Deterministic real embeddings: one dimension per vocabulary word.

    A text embeds to the L2-normalised sum of one-hot word vectors, so cosine
    similarity is exactly word overlap — nearest-neighbour ranking is knowable
    and assertable, unlike a mock that returns constant vectors.
    """

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
def pg_conn_string():
    import psycopg

    container = (
        _container_mod.DockerContainer(_PG_IMAGE)
        .with_env("POSTGRES_USER", "test")
        .with_env("POSTGRES_PASSWORD", "test")
        .with_env("POSTGRES_DB", "test")
        .with_exposed_ports(5432)
    )
    with container as pg:
        host = pg.get_container_host_ip()
        port = pg.get_exposed_port(5432)
        url = f"postgresql://test:test@{host}:{port}/test"
        # The exposed TCP port only accepts connections once the real server is
        # up (init runs on a unix socket), so a successful connect is the reliable
        # readiness signal — more robust than log scraping.
        last_err: Exception | None = None
        for _ in range(60):
            try:
                psycopg.connect(url).close()
                break
            except Exception as exc:
                last_err = exc
                time.sleep(1)
        else:
            raise RuntimeError(f"Postgres did not become ready: {last_err}")
        yield url


def _store(conn_string: str, table: str, strategy=DistanceStrategy.COSINE) -> PGVectorStore:
    return PGVectorStore(
        embedding_backend=KeywordEmbeddings(_VOCAB),
        connection_string=conn_string,
        table_name=table,
        distance_strategy=strategy,
    )


@pytest.mark.asyncio
async def test_add_and_search_returns_true_nearest_neighbor(pg_conn_string):
    store = _store(pg_conn_string, "docs_rank")
    await store.add(
        ["apple fruit red", "banana fruit yellow", "car vehicle fast"],
        [{"kind": "fruit"}, {"kind": "fruit"}, {"kind": "auto"}],
    )
    results = await store.search("apple", top_k=3)

    assert results, "expected non-empty results from a real pgvector query"
    assert results[0]["text"] == "apple fruit red"  # exact nearest neighbour
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)  # ranked by similarity
    assert 0.0 <= results[0]["score"] <= 1.0001
    assert results[0]["metadata"] == {"kind": "fruit"}  # JSONB round-trips to a dict


@pytest.mark.asyncio
async def test_metadata_filter(pg_conn_string):
    store = _store(pg_conn_string, "docs_filter")
    await store.add(
        ["apple fruit", "car vehicle"],
        [{"kind": "fruit"}, {"kind": "auto"}],
    )
    results = await store.search("apple car", top_k=5, metadata_filter={"kind": "auto"})
    assert [r["text"] for r in results] == ["car vehicle"]


@pytest.mark.asyncio
async def test_top_k_limits_results(pg_conn_string):
    store = _store(pg_conn_string, "docs_topk")
    await store.add([f"apple fruit {w}" for w in ("red", "yellow", "cherry", "banana")])
    results = await store.search("apple", top_k=2)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_search_before_add_returns_empty(pg_conn_string):
    store = _store(pg_conn_string, "docs_empty_first")
    assert await store.search("apple") == []


@pytest.mark.asyncio
async def test_persistence_across_reconnect(pg_conn_string):
    writer = _store(pg_conn_string, "docs_persist")
    await writer.add(["apple fruit red"], [{"kind": "fruit"}])

    # A fresh store instance (new connection) must see the persisted rows —
    # proves the write actually committed, not just lived in a rolled-back txn.
    reader = _store(pg_conn_string, "docs_persist")
    results = await reader.search("apple", top_k=1)
    assert results[0]["text"] == "apple fruit red"


@pytest.mark.asyncio
async def test_add_empty_is_noop(pg_conn_string):
    store = _store(pg_conn_string, "docs_noop")
    await store.add([])  # must not raise


@pytest.mark.asyncio
async def test_metadata_length_mismatch_raises(pg_conn_string):
    store = _store(pg_conn_string, "docs_mismatch")
    with pytest.raises(ValueError):
        await store.add(["a", "b"], [{"only": "one"}])
