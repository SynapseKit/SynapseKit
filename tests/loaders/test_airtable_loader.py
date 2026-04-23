from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from synapsekit.loaders import Document
from synapsekit.loaders.airtable import AirtableLoader


def test_init_requires_api_key() -> None:
    with pytest.raises(ValueError, match="api_key must be provided"):
        AirtableLoader(api_key="", base_id="base", table_name="table")


def test_init_requires_base_id() -> None:
    with pytest.raises(ValueError, match="base_id must be provided"):
        AirtableLoader(api_key="key", base_id="", table_name="table")


def test_init_requires_table_name() -> None:
    with pytest.raises(ValueError, match="table_name must be provided"):
        AirtableLoader(api_key="key", base_id="base", table_name="")


def test_load_missing_pyairtable_import_error() -> None:
    with patch.dict("sys.modules", {"pyairtable": None}):
        loader = AirtableLoader(api_key="key", base_id="base", table_name="table")
        with pytest.raises(ImportError, match=r"synapsekit\[airtable\]"):
            loader.load()


@patch.dict("sys.modules", {"pyairtable": MagicMock()})
def test_load_uses_text_fields_and_builds_metadata() -> None:
    import sys

    mock_table_class = sys.modules["pyairtable"].Table
    mock_table = MagicMock()
    mock_table_class.return_value = mock_table
    mock_table.all.return_value = [
        {
            "id": "rec1",
            "fields": {"title": "Hello", "body": "World", "count": 3},
            "createdTime": "2026-01-01T00:00:00.000Z",
        }
    ]

    loader = AirtableLoader(
        api_key="key",
        base_id="app123",
        table_name="Articles",
        text_fields=["title", "body"],
    )
    docs = loader.load()

    mock_table_class.assert_called_once_with("key", "app123", "Articles")
    mock_table.all.assert_called_once_with()
    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].text == "Hello World"
    assert docs[0].metadata["source"] == "airtable"
    assert docs[0].metadata["record_id"] == "rec1"
    assert docs[0].metadata["created_time"] == "2026-01-01T00:00:00.000Z"
    assert docs[0].metadata["title"] == "Hello"
    assert docs[0].metadata["body"] == "World"
    assert docs[0].metadata["count"] == 3


@patch.dict("sys.modules", {"pyairtable": MagicMock()})
def test_load_uses_limit_via_max_records() -> None:
    import sys

    mock_table_class = sys.modules["pyairtable"].Table
    mock_table = MagicMock()
    mock_table_class.return_value = mock_table
    mock_table.all.return_value = []

    loader = AirtableLoader(api_key="key", base_id="base", table_name="table", limit=5)
    docs = loader.load()

    assert docs == []
    mock_table.all.assert_called_once_with(max_records=5)


@patch.dict("sys.modules", {"pyairtable": MagicMock()})
def test_load_skips_empty_records() -> None:
    import sys

    mock_table_class = sys.modules["pyairtable"].Table
    mock_table = MagicMock()
    mock_table_class.return_value = mock_table
    mock_table.all.return_value = [
        {"id": "rec1", "fields": {"title": "", "body": None}, "createdTime": "2026-01-01"},
        {"id": "rec2", "fields": {"title": "Keep me"}, "createdTime": "2026-01-02"},
    ]

    loader = AirtableLoader(api_key="key", base_id="base", table_name="table")
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].metadata["record_id"] == "rec2"
    assert docs[0].text == "Keep me"


@patch.dict("sys.modules", {"pyairtable": MagicMock()})
def test_load_wraps_api_failures() -> None:
    import sys

    mock_table_class = sys.modules["pyairtable"].Table
    mock_table = MagicMock()
    mock_table_class.return_value = mock_table
    mock_table.all.side_effect = Exception("boom")

    loader = AirtableLoader(api_key="key", base_id="base", table_name="table")

    with pytest.raises(RuntimeError, match="Failed to load Airtable records"):
        loader.load()


@patch.dict("sys.modules", {"pyairtable": MagicMock()})
def test_aload_runs() -> None:
    import sys

    mock_table_class = sys.modules["pyairtable"].Table
    mock_table = MagicMock()
    mock_table_class.return_value = mock_table
    mock_table.all.return_value = [
        {"id": "rec1", "fields": {"name": "Alice"}, "createdTime": "2026-01-01"}
    ]

    loader = AirtableLoader(api_key="key", base_id="base", table_name="table")
    docs = asyncio.run(loader.aload())

    assert len(docs) == 1
    assert docs[0].text == "Alice"
