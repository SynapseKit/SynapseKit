from __future__ import annotations

import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest


class _Schema:
    def __init__(self) -> None:
        self.fields: list[tuple[tuple, dict]] = []

    def add_field(self, *args, **kwargs):
        self.fields.append((args, kwargs))


def _fake_pymilvus(client: MagicMock):
    module = types.ModuleType("pymilvus")
    module.MilvusClient = MagicMock(return_value=client)
    module.DataType = SimpleNamespace(INT64="INT64", VARCHAR="VARCHAR", FLOAT_VECTOR="FLOAT_VECTOR")
    return module


def _embedding_backend():
    backend = MagicMock()
    backend.dimension = 3
    backend.embed = AsyncMock(
        return_value=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    )
    backend.embed_one = AsyncMock(return_value=np.array([1.0, 0.0, 0.0], dtype=np.float32))
    return backend


@pytest.mark.asyncio
async def test_milvus_store_creates_collection_inserts_rows_and_searches_with_metadata_filter(
    monkeypatch,
):
    client = MagicMock()
    client.has_collection.side_effect = [False, True]
    schema = _Schema()
    client.create_schema.return_value = schema
    client.search.return_value = [
        [
            SimpleNamespace(
                entity={
                    "id": 1,
                    "text": "alpha",
                    "embedding": [1.0, 0.0, 0.0],
                    "source": "notes",
                    "lang": "en",
                },
                distance=0.91,
            )
        ]
    ]

    monkeypatch.setitem(__import__("sys").modules, "pymilvus", _fake_pymilvus(client))

    from synapsekit.retrieval.milvus import MilvusIndexType, MilvusVectorStore

    store = MilvusVectorStore(
        _embedding_backend(),
        collection_name="docs",
        index_type=MilvusIndexType.HNSW,
    )

    client.create_collection.assert_not_called()
    client.create_index.assert_not_called()
    client.load_collection.assert_not_called()

    await store.add(
        ["alpha", "beta"],
        metadata=[{"source": "notes", "lang": "en"}, {"source": "blog", "lang": "fr"}],
    )

    client.create_collection.assert_called_once()
    client.create_index.assert_called_once_with(
        collection_name="docs",
        field_name="embedding",
        index_params={
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        },
    )
    client.load_collection.assert_called_once_with(collection_name="docs")
    assert len(schema.fields) == 3

    client.insert.assert_called_once_with(
        collection_name="docs",
        data=[
            {"text": "alpha", "embedding": [1.0, 0.0, 0.0], "source": "notes", "lang": "en"},
            {"text": "beta", "embedding": [0.0, 1.0, 0.0], "source": "blog", "lang": "fr"},
        ],
    )

    results = await store.search(
        "alpha", top_k=3, metadata_filter={"source": "notes", "lang": "en"}
    )

    client.search.assert_called_once_with(
        collection_name="docs",
        data=[[1.0, 0.0, 0.0]],
        limit=3,
        filter='source == "notes" and lang == "en"',
        output_fields=["text"],
        search_params={"metric_type": "COSINE", "params": {"ef": 64}},
    )
    assert results == [
        {
            "text": "alpha",
            "score": 0.91,
            "metadata": {"source": "notes", "lang": "en"},
        }
    ]


@pytest.mark.asyncio
async def test_milvus_store_rejects_metadata_length_mismatch(monkeypatch):
    client = MagicMock()
    client.has_collection.return_value = True
    monkeypatch.setitem(__import__("sys").modules, "pymilvus", _fake_pymilvus(client))

    from synapsekit.retrieval.milvus import MilvusVectorStore

    store = MilvusVectorStore(_embedding_backend(), collection_name="docs")

    with pytest.raises(ValueError, match="metadata must match texts length"):
        await store.add(["alpha"], metadata=[{"source": "notes"}, {"source": "extra"}])
