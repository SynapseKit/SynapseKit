from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest

from synapsekit.loaders.base import Document
from synapsekit.loaders.snowflake import SnowflakeLoader


def _patch_snowflake_connect(connect_mock: MagicMock):
    snowflake_mod = types.ModuleType("snowflake")
    connector_mod = types.ModuleType("snowflake.connector")
    connector_mod.connect = connect_mock
    snowflake_mod.connector = connector_mod
    return patch.dict(
        "sys.modules",
        {
            "snowflake": snowflake_mod,
            "snowflake.connector": connector_mod,
        },
    )


def test_basic_query_returns_documents() -> None:
    mock_cur = MagicMock()
    mock_cur.description = [("id",), ("title",), ("body",)]
    mock_cur.fetchall.return_value = [
        (1, "Doc One", "Body One"),
        (2, "Doc Two", "Body Two"),
    ]

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cur

    connect_mock = MagicMock(return_value=mock_conn)
    with _patch_snowflake_connect(connect_mock):
        loader = SnowflakeLoader(
            account="acct",
            user="user",
            password="pass",
            query="SELECT id, title, body FROM docs",
        )
        docs = loader.load()

    assert len(docs) == 2
    assert all(isinstance(d, Document) for d in docs)
    assert docs[0].text == "1 Doc One Body One"
    assert docs[0].metadata["source"] == "snowflake"
    assert docs[0].metadata["query"] == "SELECT id, title, body FROM docs"
    assert docs[0].metadata["id"] == 1
    mock_cur.execute.assert_called_once_with("SELECT id, title, body FROM docs")
    mock_cur.close.assert_called_once()
    mock_conn.close.assert_called_once()


def test_text_fields_build_text_correctly() -> None:
    mock_cur = MagicMock()
    mock_cur.description = [("title",), ("body",), ("extra",)]
    mock_cur.fetchall.return_value = [("My Title", "My Body", "ignored")]

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cur

    connect_mock = MagicMock(return_value=mock_conn)
    with _patch_snowflake_connect(connect_mock):
        loader = SnowflakeLoader(
            account="acct",
            user="user",
            password="pass",
            query="SELECT title, body, extra FROM docs",
            text_fields=["title", "body"],
        )
        docs = loader.load()

    assert len(docs) == 1
    assert docs[0].text == "My Title My Body"


def test_empty_results_returns_empty_list() -> None:
    mock_cur = MagicMock()
    mock_cur.description = [("id",), ("title",)]
    mock_cur.fetchall.return_value = []

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cur

    connect_mock = MagicMock(return_value=mock_conn)
    with _patch_snowflake_connect(connect_mock):
        loader = SnowflakeLoader(
            account="acct",
            user="user",
            password="pass",
            query="SELECT id, title FROM docs",
        )
        docs = loader.load()

    assert docs == []


def test_limit_restricts_row_count() -> None:
    mock_cur = MagicMock()
    mock_cur.description = [("id",), ("title",)]
    mock_cur.fetchall.return_value = [
        (1, "One"),
        (2, "Two"),
        (3, "Three"),
    ]

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cur

    connect_mock = MagicMock(return_value=mock_conn)
    with _patch_snowflake_connect(connect_mock):
        loader = SnowflakeLoader(
            account="acct",
            user="user",
            password="pass",
            query="SELECT id, title FROM docs",
            limit=2,
        )
        docs = loader.load()

    assert len(docs) == 2
    assert docs[0].metadata["id"] == 1
    assert docs[1].metadata["id"] == 2


def test_connection_error_raises_runtime_error() -> None:
    connect_mock = MagicMock(side_effect=Exception("bad connection"))
    with _patch_snowflake_connect(connect_mock):
        loader = SnowflakeLoader(
            account="acct",
            user="user",
            password="pass",
            query="SELECT 1",
        )
        with pytest.raises(RuntimeError, match="Snowflake query failed: bad connection"):
            loader.load()
