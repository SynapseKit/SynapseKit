from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from synapsekit.loaders import Document
from synapsekit.loaders.elasticsearch import ElasticsearchLoader

# ---------------------------------------------------------------------------
# Initialisation validation
# ---------------------------------------------------------------------------


def test_init_requires_url() -> None:
    with pytest.raises(ValueError, match="url must be provided"):
        ElasticsearchLoader(url="", index="my-index")


def test_init_requires_index() -> None:
    with pytest.raises(ValueError, match="index must be provided"):
        ElasticsearchLoader(url="http://localhost:9200", index="")


def test_init_defaults() -> None:
    loader = ElasticsearchLoader(url="http://localhost:9200", index="my-index")
    assert loader._text_fields == ["content"]
    assert loader._query is None
    assert loader._limit is None


# ---------------------------------------------------------------------------
# Missing dependency
# ---------------------------------------------------------------------------


def test_load_import_error_missing_elasticsearch() -> None:
    with patch.dict("sys.modules", {"elasticsearch": None}):
        loader = ElasticsearchLoader(url="http://localhost:9200", index="my-index")
        with pytest.raises(ImportError, match="elasticsearch required"):
            loader.load()


# ---------------------------------------------------------------------------
# Normal load
# ---------------------------------------------------------------------------


@patch.dict("sys.modules", {"elasticsearch": MagicMock()})
def test_load_returns_documents() -> None:
    import sys

    mock_es_class = sys.modules["elasticsearch"].Elasticsearch
    mock_client = MagicMock()
    mock_es_class.return_value = mock_client

    mock_client.search.return_value = {
        "hits": {
            "hits": [
                {"_id": "1", "_source": {"content": "Hello world"}},
                {"_id": "2", "_source": {"content": "Elasticsearch rocks"}},
            ]
        }
    }

    loader = ElasticsearchLoader(url="http://localhost:9200", index="articles")
    docs = loader.load()

    assert len(docs) == 2
    assert all(isinstance(doc, Document) for doc in docs)
    assert docs[0].text == "Hello world"
    assert docs[1].text == "Elasticsearch rocks"


@patch.dict("sys.modules", {"elasticsearch": MagicMock()})
def test_load_metadata_correctness() -> None:
    import sys

    mock_es_class = sys.modules["elasticsearch"].Elasticsearch
    mock_client = MagicMock()
    mock_es_class.return_value = mock_client

    mock_client.search.return_value = {
        "hits": {
            "hits": [
                {"_id": "abc-123", "_source": {"content": "Test document"}},
            ]
        }
    }

    loader = ElasticsearchLoader(url="http://localhost:9200", index="docs")
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].metadata["source"] == "elasticsearch"
    assert docs[0].metadata["index"] == "docs"
    assert docs[0].metadata["id"] == "abc-123"


@patch.dict("sys.modules", {"elasticsearch": MagicMock()})
def test_load_multiple_text_fields() -> None:
    import sys

    mock_es_class = sys.modules["elasticsearch"].Elasticsearch
    mock_client = MagicMock()
    mock_es_class.return_value = mock_client

    mock_client.search.return_value = {
        "hits": {
            "hits": [
                {"_id": "1", "_source": {"title": "My Title", "body": "My Body"}},
            ]
        }
    }

    loader = ElasticsearchLoader(
        url="http://localhost:9200",
        index="articles",
        text_fields=["title", "body"],
    )
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].text == "My Title My Body"


# ---------------------------------------------------------------------------
# Query forwarding
# ---------------------------------------------------------------------------


@patch.dict("sys.modules", {"elasticsearch": MagicMock()})
def test_load_passes_custom_query() -> None:
    import sys

    mock_es_class = sys.modules["elasticsearch"].Elasticsearch
    mock_client = MagicMock()
    mock_es_class.return_value = mock_client

    mock_client.search.return_value = {"hits": {"hits": []}}

    custom_query = {"term": {"status": "published"}}
    loader = ElasticsearchLoader(
        url="http://localhost:9200",
        index="posts",
        query=custom_query,
    )
    loader.load()

    mock_client.search.assert_called_once_with(
        index="posts",
        query=custom_query,
        size=100,
    )


@patch.dict("sys.modules", {"elasticsearch": MagicMock()})
def test_load_defaults_to_match_all_query() -> None:
    import sys

    mock_es_class = sys.modules["elasticsearch"].Elasticsearch
    mock_client = MagicMock()
    mock_es_class.return_value = mock_client

    mock_client.search.return_value = {"hits": {"hits": []}}

    loader = ElasticsearchLoader(url="http://localhost:9200", index="logs")
    loader.load()

    mock_client.search.assert_called_once_with(
        index="logs",
        query={"match_all": {}},
        size=100,
    )


@patch.dict("sys.modules", {"elasticsearch": MagicMock()})
def test_load_respects_limit() -> None:
    import sys

    mock_es_class = sys.modules["elasticsearch"].Elasticsearch
    mock_client = MagicMock()
    mock_es_class.return_value = mock_client

    mock_client.search.return_value = {"hits": {"hits": []}}

    loader = ElasticsearchLoader(url="http://localhost:9200", index="logs", limit=10)
    loader.load()

    mock_client.search.assert_called_once_with(
        index="logs",
        query={"match_all": {}},
        size=10,
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@patch.dict("sys.modules", {"elasticsearch": MagicMock()})
def test_load_empty_results() -> None:
    import sys

    mock_es_class = sys.modules["elasticsearch"].Elasticsearch
    mock_client = MagicMock()
    mock_es_class.return_value = mock_client

    mock_client.search.return_value = {"hits": {"hits": []}}

    loader = ElasticsearchLoader(url="http://localhost:9200", index="empty-index")
    docs = loader.load()

    assert docs == []


@patch.dict("sys.modules", {"elasticsearch": MagicMock()})
def test_load_skips_hits_with_empty_text() -> None:
    import sys

    mock_es_class = sys.modules["elasticsearch"].Elasticsearch
    mock_client = MagicMock()
    mock_es_class.return_value = mock_client

    mock_client.search.return_value = {
        "hits": {
            "hits": [
                {"_id": "1", "_source": {"content": ""}},
                {"_id": "2", "_source": {}},
                {"_id": "3", "_source": {"content": "Valid content"}},
            ]
        }
    }

    loader = ElasticsearchLoader(url="http://localhost:9200", index="mixed")
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].text == "Valid content"
    assert docs[0].metadata["id"] == "3"


@patch.dict("sys.modules", {"elasticsearch": MagicMock()})
def test_load_skips_none_field_values() -> None:
    import sys

    mock_es_class = sys.modules["elasticsearch"].Elasticsearch
    mock_client = MagicMock()
    mock_es_class.return_value = mock_client

    mock_client.search.return_value = {
        "hits": {
            "hits": [
                {"_id": "1", "_source": {"title": None, "body": "Some body text"}},
            ]
        }
    }

    loader = ElasticsearchLoader(
        url="http://localhost:9200",
        index="articles",
        text_fields=["title", "body"],
    )
    docs = loader.load()

    # title is None so only body goes into text — no "None" strings
    assert len(docs) == 1
    assert docs[0].text == "Some body text"
    assert "None" not in docs[0].text


@patch.dict("sys.modules", {"elasticsearch": MagicMock()})
def test_load_missing_source_is_skipped() -> None:
    import sys

    mock_es_class = sys.modules["elasticsearch"].Elasticsearch
    mock_client = MagicMock()
    mock_es_class.return_value = mock_client

    mock_client.search.return_value = {
        "hits": {
            "hits": [
                {"_id": "1"},  # no _source key
                {"_id": "2", "_source": {"content": "Good doc"}},
            ]
        }
    }

    loader = ElasticsearchLoader(url="http://localhost:9200", index="partial")
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].text == "Good doc"


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


@patch.dict("sys.modules", {"elasticsearch": MagicMock()})
def test_aload() -> None:
    import sys

    mock_es_class = sys.modules["elasticsearch"].Elasticsearch
    mock_client = MagicMock()
    mock_es_class.return_value = mock_client

    mock_client.search.return_value = {
        "hits": {
            "hits": [
                {"_id": "1", "_source": {"content": "Async test"}},
            ]
        }
    }

    loader = ElasticsearchLoader(url="http://localhost:9200", index="async-index")
    docs = asyncio.run(loader.aload())

    assert len(docs) == 1
    assert docs[0].text == "Async test"
