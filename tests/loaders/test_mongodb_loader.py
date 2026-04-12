from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from synapsekit.loaders import Document
from synapsekit.loaders.mongodb import MongoDBLoader


def test_init_requires_connection_string() -> None:
    with pytest.raises(ValueError, match="connection_string must be provided"):
        MongoDBLoader(connection_string="", database="db", collection="col")


def test_init_requires_database() -> None:
    with pytest.raises(ValueError, match="database must be provided"):
        MongoDBLoader(connection_string="mongodb://localhost:27017", database="", collection="col")


def test_init_requires_collection() -> None:
    with pytest.raises(ValueError, match="collection must be provided"):
        MongoDBLoader(connection_string="mongodb://localhost:27017", database="db", collection="")


def test_load_import_error_missing_pymongo() -> None:
    with patch.dict("sys.modules", {"pymongo": None}):
        loader = MongoDBLoader(
            connection_string="mongodb://localhost:27017",
            database="analytics",
            collection="events",
        )
        with pytest.raises(ImportError, match="pymongo required"):
            loader.load()


@patch.dict("sys.modules", {"pymongo": MagicMock()})
def test_load_with_default_text_and_metadata() -> None:
    import sys

    mock_mongo_client = sys.modules["pymongo"].MongoClient
    mock_client = MagicMock()
    mock_mongo_client.return_value = mock_client

    rows = [
        {"_id": "a1", "title": "First", "content": "Hello", "author": "Alice"},
        {"_id": "a2", "title": "Second", "content": "World", "author": "Bob"},
    ]

    mock_collection = MagicMock()
    mock_collection.find.return_value = rows

    mock_db = MagicMock()
    mock_db.__getitem__.return_value = mock_collection
    mock_client.__getitem__.return_value = mock_db

    loader = MongoDBLoader(
        connection_string="mongodb://localhost:27017",
        database="analytics",
        collection="articles",
        query_filter={"author": "Alice"},
    )

    docs = loader.load()

    assert len(docs) == 2
    assert all(isinstance(doc, Document) for doc in docs)

    assert "title: First" in docs[0].text
    assert "content: Hello" in docs[0].text

    assert docs[0].metadata["source"] == "mongodb"
    assert docs[0].metadata["database"] == "analytics"
    assert docs[0].metadata["collection"] == "articles"
    assert docs[0].metadata["row"] == 0
    assert docs[0].metadata["query"] == {"author": "Alice"}
    assert docs[0].metadata["author"] == "Alice"

    mock_collection.find.assert_called_once_with({"author": "Alice"})
    mock_client.close.assert_called_once()


@patch.dict("sys.modules", {"pymongo": MagicMock()})
def test_load_with_text_fields_and_metadata_fields() -> None:
    import sys

    mock_mongo_client = sys.modules["pymongo"].MongoClient
    mock_client = MagicMock()
    mock_mongo_client.return_value = mock_client

    rows = [
        {
            "_id": "x1",
            "title": "Mongo Intro",
            "body": "A beginner guide",
            "author": "Carol",
            "tags": ["db", "mongo"],
        }
    ]

    mock_collection = MagicMock()
    mock_collection.find.return_value = rows

    mock_db = MagicMock()
    mock_db.__getitem__.return_value = mock_collection
    mock_client.__getitem__.return_value = mock_db

    loader = MongoDBLoader(
        connection_string="mongodb://localhost:27017",
        database="content",
        collection="posts",
        text_fields=["title", "body"],
        metadata_fields=["_id", "author"],
    )

    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].text == "Mongo Intro\nA beginner guide"

    assert docs[0].metadata["source"] == "mongodb"
    assert docs[0].metadata["database"] == "content"
    assert docs[0].metadata["collection"] == "posts"
    assert docs[0].metadata["_id"] == "x1"
    assert docs[0].metadata["author"] == "Carol"
    assert "tags" not in docs[0].metadata

    mock_collection.find.assert_called_once_with({}, {"_id": 1, "title": 1, "body": 1, "author": 1})


@patch.dict("sys.modules", {"pymongo": MagicMock()})
def test_load_empty_results() -> None:
    import sys

    mock_mongo_client = sys.modules["pymongo"].MongoClient
    mock_client = MagicMock()
    mock_mongo_client.return_value = mock_client

    mock_collection = MagicMock()
    mock_collection.find.return_value = []

    mock_db = MagicMock()
    mock_db.__getitem__.return_value = mock_collection
    mock_client.__getitem__.return_value = mock_db

    loader = MongoDBLoader(
        connection_string="mongodb://localhost:27017",
        database="content",
        collection="posts",
    )

    docs = loader.load()
    assert docs == []


def test_query_filter_defensive_copy() -> None:
    query = {"status": "active"}
    loader = MongoDBLoader(
        connection_string="mongodb://localhost:27017",
        database="content",
        collection="posts",
        query_filter=query,
    )
    query["status"] = "inactive"

    assert loader._query_filter == {"status": "active"}


@patch.dict("sys.modules", {"pymongo": MagicMock()})
def test_text_fields_none_values_are_empty_strings() -> None:
    import sys

    mock_mongo_client = sys.modules["pymongo"].MongoClient
    mock_client = MagicMock()
    mock_mongo_client.return_value = mock_client

    rows = [{"_id": "1", "title": None, "body": "content"}]

    mock_collection = MagicMock()
    mock_collection.find.return_value = rows

    mock_db = MagicMock()
    mock_db.__getitem__.return_value = mock_collection
    mock_client.__getitem__.return_value = mock_db

    loader = MongoDBLoader(
        connection_string="mongodb://localhost:27017",
        database="db",
        collection="col",
        text_fields=["title", "body"],
    )

    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].text == "\ncontent"


@patch.dict("sys.modules", {"pymongo": MagicMock()})
def test_aload() -> None:
    import sys

    mock_mongo_client = sys.modules["pymongo"].MongoClient
    mock_client = MagicMock()
    mock_mongo_client.return_value = mock_client

    rows = [{"_id": "1", "text": "hello"}]

    mock_collection = MagicMock()
    mock_collection.find.return_value = rows

    mock_db = MagicMock()
    mock_db.__getitem__.return_value = mock_collection
    mock_client.__getitem__.return_value = mock_db

    loader = MongoDBLoader(
        connection_string="mongodb://localhost:27017",
        database="db",
        collection="col",
        text_fields=["text"],
    )

    docs = asyncio.run(loader.aload())

    assert len(docs) == 1
    assert docs[0].text == "hello"
