from __future__ import annotations

import types
from unittest.mock import MagicMock

import numpy as np
import pytest


def _make_embeddings(dim: int = 4):
    mock = MagicMock()

    async def embed(texts):
        vecs = []
        for i, _text in enumerate(texts):
            vecs.append(np.eye(dim, dtype=np.float32)[i % dim])
        return np.array(vecs, dtype=np.float32)

    async def embed_one(_text):
        return np.eye(dim, dtype=np.float32)[0]

    mock.embed = embed
    mock.embed_one = embed_one
    return mock


def _fake_lancedb(db: MagicMock):
    module = types.ModuleType("lancedb")
    module.connect = MagicMock(return_value=db)
    return module


@pytest.mark.asyncio
async def test_lancedb_store_adds_searches_and_creates_fts_index(monkeypatch):
    builder = MagicMock()
    builder.where.return_value = builder
    builder.limit.return_value = builder
    builder.to_list.return_value = [
        {
            "text": "alpha",
            "_score": 0.97,
            "source": "notes",
            "lang": "en",
        }
    ]

    table = MagicMock()
    table.create_fts_index.return_value = None
    table.search.return_value = builder
    table.add.return_value = None

    db = MagicMock()
    db.open_table.return_value = table
    db.create_table.return_value = table

    monkeypatch.setitem(__import__("sys").modules, "lancedb", _fake_lancedb(db))

    from synapsekit.retrieval.lancedb import LanceDBVectorStore

    store = LanceDBVectorStore(_make_embeddings(), uri="/tmp/lancedb", table_name="docs")

    await store.add(
        ["alpha", "beta"],
        metadata=[{"source": "notes", "lang": "en"}, {"source": "blog", "lang": "fr"}],
    )

    db.open_table.assert_called_once_with("docs")
    db.create_table.assert_not_called()
    table.create_fts_index.assert_called_once_with("text")
    table.add.assert_called_once_with(
        [
            {"text": "alpha", "embedding": [1.0, 0.0, 0.0, 0.0], "source": "notes", "lang": "en"},
            {"text": "beta", "embedding": [0.0, 1.0, 0.0, 0.0], "source": "blog", "lang": "fr"},
        ]
    )

    results = await store.search(
        "alpha", top_k=3, metadata_filter={"source": "notes", "lang": "en"}
    )

    table.search.assert_called_once_with("alpha")
    builder.where.assert_called_once_with('source == "notes" and lang == "en"')
    builder.limit.assert_called_once_with(3)
    builder.to_list.assert_called_once()
    assert results == [
        {
            "text": "alpha",
            "score": 0.97,
            "metadata": {"source": "notes", "lang": "en"},
        }
    ]


@pytest.mark.asyncio
async def test_lancedb_store_creates_table_when_missing(monkeypatch):
    builder = MagicMock()
    builder.limit.return_value = builder
    builder.to_list.return_value = []

    table = MagicMock()
    table.create_fts_index.return_value = None
    table.search.return_value = builder
    table.add.return_value = None

    db = MagicMock()
    db.open_table.side_effect = Exception("missing")
    db.create_table.return_value = table

    monkeypatch.setitem(__import__("sys").modules, "lancedb", _fake_lancedb(db))

    from synapsekit.retrieval.lancedb import LanceDBVectorStore

    store = LanceDBVectorStore(_make_embeddings(), uri="/tmp/lancedb", table_name="docs")

    await store.add(["alpha"])

    db.create_table.assert_called_once()
    table.create_fts_index.assert_called_once_with("text")
    table.add.assert_not_called()


@pytest.mark.asyncio
async def test_lancedb_store_rejects_metadata_length_mismatch(monkeypatch):
    db = MagicMock()
    db.open_table.side_effect = Exception("missing")
    monkeypatch.setitem(__import__("sys").modules, "lancedb", _fake_lancedb(db))

    from synapsekit.retrieval.lancedb import LanceDBVectorStore

    store = LanceDBVectorStore(_make_embeddings(), uri="/tmp/lancedb", table_name="docs")

    with pytest.raises(ValueError, match="metadata must match texts length"):
        await store.add(["alpha"], metadata=[{"source": "notes"}, {"source": "extra"}])
