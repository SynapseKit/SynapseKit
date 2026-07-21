"""Real MilvusVectorStore integration tests via Milvus Lite (embedded real engine).

Milvus standalone needs etcd + minio (a heavy 3-container deployment); Milvus Lite
runs the same real Milvus engine embedded against a local ``.db`` file, so this is
real vector search (not a mock), just without the ops overhead. Exercises true
nearest-neighbour ranking, metadata filtering, top_k, persistence across reopen,
empty/negative cases, and the async-first contract. Part of #829.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

pytest.importorskip("pymilvus")
pytest.importorskip("milvus_lite")

from synapsekit.retrieval.milvus import MilvusVectorStore

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


def _store(tmp_path, collection: str = "docs") -> MilvusVectorStore:
    return MilvusVectorStore(
        embedding_backend=KeywordEmbeddings(_VOCAB),
        uri=str(tmp_path / "milvus.db"),
        collection_name=collection,
    )


@pytest.mark.asyncio
async def test_add_and_search_returns_true_nearest_neighbor(tmp_path):
    store = _store(tmp_path)
    await store.add(
        ["apple fruit red", "banana fruit yellow", "car vehicle fast"],
        [{"kind": "fruit"}, {"kind": "fruit"}, {"kind": "auto"}],
    )
    results = await store.search("apple", top_k=3)
    assert results, "expected non-empty results from a real Milvus query"
    assert results[0]["text"] == "apple fruit red"
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)  # COSINE similarity -> descending
    assert results[0]["metadata"] == {"kind": "fruit"}


@pytest.mark.asyncio
async def test_metadata_filter(tmp_path):
    store = _store(tmp_path)
    await store.add(["apple fruit", "car vehicle"], [{"kind": "fruit"}, {"kind": "auto"}])
    results = await store.search("apple car", top_k=5, metadata_filter={"kind": "auto"})
    assert [r["text"] for r in results] == ["car vehicle"]


@pytest.mark.asyncio
async def test_top_k_limits_results(tmp_path):
    store = _store(tmp_path)
    await store.add([f"apple fruit {w}" for w in ("red", "yellow", "cherry", "banana")])
    results = await store.search("apple", top_k=2)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_search_before_add_returns_empty(tmp_path):
    store = _store(tmp_path)
    assert await store.search("apple") == []


@pytest.mark.asyncio
async def test_persistence_across_reopen(tmp_path):
    writer = _store(tmp_path)
    await writer.add(["apple fruit red"], [{"kind": "fruit"}])

    # A second store on the same on-disk db must see the persisted vector.
    reader = _store(tmp_path)
    results = await reader.search("apple", top_k=1)
    assert results[0]["text"] == "apple fruit red"


@pytest.mark.asyncio
async def test_add_empty_is_noop(tmp_path):
    store = _store(tmp_path)
    await store.add([])


@pytest.mark.asyncio
async def test_metadata_length_mismatch_raises(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(ValueError):
        await store.add(["a", "b"], [{"only": "one"}])


def test_async_api_is_coroutine(tmp_path):
    # SynapseKit is async-first: the public IO surface must stay coroutines.
    store = _store(tmp_path)
    assert inspect.iscoroutinefunction(store.add)
    assert inspect.iscoroutinefunction(store.search)
