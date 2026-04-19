from __future__ import annotations

import asyncio
import importlib
from unittest.mock import MagicMock, patch

import pytest

from synapsekit.loaders import BigQueryLoader, Document


def test_init_requires_query_or_table() -> None:
    with pytest.raises(ValueError, match="Either query or table must be provided"):
        BigQueryLoader()


def test_init_rejects_query_and_table_together() -> None:
    with pytest.raises(ValueError, match="Provide either query or table, not both"):
        BigQueryLoader(query="SELECT 1", table="events", dataset="analytics")


def test_init_requires_dataset_for_unqualified_table() -> None:
    with pytest.raises(ValueError, match="dataset must be provided"):
        BigQueryLoader(table="events")


def test_load_import_error_missing_google_cloud_bigquery() -> None:
    with patch.dict("sys.modules", {"google": None}):
        loader = BigQueryLoader(query="SELECT 1")
        with pytest.raises(ImportError, match="google-cloud-bigquery required"):
            loader.load()


def test_load_with_query_default_text_and_metadata() -> None:
    rows = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]

    mock_job = MagicMock()
    mock_job.result.return_value = rows

    mock_client = MagicMock()
    mock_client.query.return_value = mock_job

    query = "SELECT id, name FROM users"
    loader = BigQueryLoader(query=query, project="demo-project", client=mock_client)

    docs = loader.load()

    assert len(docs) == 2
    assert all(isinstance(doc, Document) for doc in docs)
    assert "id: 1" in docs[0].text
    assert "name: Alice" in docs[0].text
    assert docs[0].metadata["source"] == "bigquery"
    assert docs[0].metadata["query"] == query
    assert docs[0].metadata["row"] == 0
    assert docs[0].metadata["project"] == "demo-project"
    assert docs[0].metadata["id"] == 1

    mock_client.query.assert_called_once_with(query)


def test_load_with_table_and_field_selection() -> None:
    rows = [{"id": 10, "body": "Hello", "status": "active"}]

    mock_job = MagicMock()
    mock_job.result.return_value = rows

    mock_client = MagicMock()
    mock_client.query.return_value = mock_job

    loader = BigQueryLoader(
        project="demo-project",
        dataset="analytics",
        table="events",
        client=mock_client,
        text_fields=["body"],
        metadata_fields=["id"],
    )

    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].text == "body: Hello"
    assert docs[0].metadata["source"] == "bigquery"
    assert docs[0].metadata["project"] == "demo-project"
    assert docs[0].metadata["dataset"] == "analytics"
    assert docs[0].metadata["table"] == "events"
    assert docs[0].metadata["id"] == 10
    assert "status" not in docs[0].metadata

    mock_client.query.assert_called_once_with("SELECT * FROM `demo-project.analytics.events`")


def test_load_with_fully_qualified_table() -> None:
    mock_job = MagicMock()
    mock_job.result.return_value = [{"id": "row-1"}]

    mock_client = MagicMock()
    mock_client.query.return_value = mock_job

    loader = BigQueryLoader(table="demo-project.analytics.events", client=mock_client)
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].metadata["table"] == "demo-project.analytics.events"
    mock_client.query.assert_called_once_with("SELECT * FROM `demo-project.analytics.events`")


def test_load_skips_non_mapping_rows() -> None:
    mock_job = MagicMock()
    mock_job.result.return_value = [
        "invalid-row",
        {"id": 1, "body": "ok"},
    ]

    mock_client = MagicMock()
    mock_client.query.return_value = mock_job

    loader = BigQueryLoader(query="SELECT id, body FROM events", client=mock_client)
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].metadata["id"] == 1


def test_aload() -> None:
    mock_job = MagicMock()
    mock_job.result.return_value = [{"name": "Taylor"}]

    mock_client = MagicMock()
    mock_client.query.return_value = mock_job

    loader = BigQueryLoader(query="SELECT name FROM users", client=mock_client)
    docs = asyncio.run(loader.aload())

    assert len(docs) == 1
    assert docs[0].text == "name: Taylor"


def test_exports_from_synapsekit_and_loaders_modules() -> None:
    import synapsekit
    from synapsekit.loaders import __all__ as loaders_all

    loaders = importlib.import_module("synapsekit.loaders")

    assert "BigQueryLoader" in synapsekit.__all__
    assert "BigQueryLoader" in loaders_all
    assert "BigQueryLoader" in loaders.__all__
    assert hasattr(synapsekit, "BigQueryLoader")
    assert hasattr(loaders, "BigQueryLoader")
