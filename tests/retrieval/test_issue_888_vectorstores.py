"""Contract tests for the vector-store expansion in issue #888."""

from __future__ import annotations

import importlib
import inspect

import pytest

_BACKENDS = (
    ("turbopuffer", "TurbopufferVectorStore"),
    ("azure_ai_search", "AzureAISearchVectorStore"),
    ("vertex_ai_vector", "VertexAIVectorStore"),
    ("singlestore_vector", "SingleStoreVectorStore"),
    ("tidb_vector", "TiDBVectorStore"),
    ("couchbase_vector", "CouchbaseVectorStore"),
    ("surrealdb_vector", "SurrealDBVectorStore"),
    ("deeplake", "DeepLakeVectorStore"),
    ("myscale_vector", "MyScaleVectorStore"),
)


@pytest.mark.parametrize(("module_name", "class_name"), _BACKENDS)
def test_issue_888_backends_expose_vector_store_contract(module_name: str, class_name: str):
    from synapsekit.retrieval.base import VectorStore

    module = importlib.import_module(f"synapsekit.retrieval.{module_name}")
    backend = getattr(module, class_name)

    assert issubclass(backend, VectorStore)
    assert inspect.iscoroutinefunction(backend.add)
    assert inspect.iscoroutinefunction(backend.search)
    assert inspect.iscoroutinefunction(backend.search_mmr)
    assert callable(backend.save)
    assert callable(backend.load)


def test_issue_888_backends_are_lazy_exports():
    import synapsekit
    import synapsekit.retrieval as retrieval

    for _, class_name in _BACKENDS:
        assert class_name in retrieval.__all__
        assert class_name in retrieval._BACKENDS
        assert class_name in synapsekit.__all__
        assert class_name in synapsekit._LAZY_IMPORTS


def test_remote_contract_mmr_and_snapshot_round_trip(tmp_path):
    from synapsekit.retrieval._remote_vector import RemoteVectorStoreSupport
    from synapsekit.retrieval.base import VectorStore

    class Store(RemoteVectorStoreSupport, VectorStore):
        def __init__(self):
            self._embeddings = _Embeddings()
            self._init_remote_state()

        async def add(self, texts, metadata=None):
            docs = self._validate_documents(texts, metadata)
            self._remember_documents(docs)

        async def search(self, query, top_k=5, metadata_filter=None):
            await self._flush_pending()
            return [
                {"text": text, "score": 1.0, "metadata": meta}
                for text, meta in self._documents[:top_k]
                if not metadata_filter
                or all(meta.get(key) == value for key, value in metadata_filter.items())
            ]

    store = Store()
    import asyncio

    asyncio.run(store.add(["alpha", "beta"], [{"kind": "a"}, {"kind": "b"}]))
    result = asyncio.run(store.search_mmr("alpha", top_k=1, fetch_k=2))
    assert result[0]["text"] == "alpha"

    path = tmp_path / "snapshot.json"
    store.save(str(path))
    restored = Store()
    restored.load(str(path))
    assert asyncio.run(restored.search("alpha", top_k=2)) == [
        {"text": "alpha", "score": 1.0, "metadata": {"kind": "a"}},
        {"text": "beta", "score": 1.0, "metadata": {"kind": "b"}},
    ]


class _Embeddings:
    async def embed(self, texts):
        return [[1.0, 0.0] if "alpha" in text else [0.0, 1.0] for text in texts]

    async def embed_one(self, text):
        return [1.0, 0.0] if "alpha" in text else [0.0, 1.0]
