"""Production-grade tests for SnowflakeLoader — LIMIT SQL regression."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from synapsekit.loaders.snowflake import SnowflakeLoader

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_snowflake_mock(rows, columns):
    """Return a fake snowflake.connector module with a pre-configured cursor."""
    mock_cur = MagicMock()
    mock_cur.fetchall.return_value = rows
    mock_cur.description = [(c,) for c in columns]
    mock_cur.execute.return_value = None
    mock_cur.close.return_value = None

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    mock_conn.close.return_value = None

    mock_connector = MagicMock()
    mock_connector.connect.return_value = mock_conn

    mock_snowflake_pkg = types.ModuleType("snowflake")
    mock_snowflake_pkg.connector = mock_connector

    return mock_snowflake_pkg, mock_connector, mock_conn, mock_cur


# ── construction validation ──────────────────────────────────────────────────


class TestSnowflakeLoaderValidation:
    def test_missing_account_raises(self):
        with pytest.raises(ValueError, match="account"):
            SnowflakeLoader(account="", user="u", password="p", query="SELECT 1")

    def test_missing_user_raises(self):
        with pytest.raises(ValueError, match="user"):
            SnowflakeLoader(account="a", user="", password="p", query="SELECT 1")

    def test_missing_password_raises(self):
        with pytest.raises(ValueError, match="password"):
            SnowflakeLoader(account="a", user="u", password="", query="SELECT 1")

    def test_missing_query_raises(self):
        with pytest.raises(ValueError, match="query"):
            SnowflakeLoader(account="a", user="u", password="p", query="")

    def test_valid_construction(self):
        loader = SnowflakeLoader(account="a", user="u", password="p", query="SELECT 1")
        assert loader is not None


# ── LIMIT SQL injection (critical regression) ────────────────────────────────


class TestSnowflakeLoaderLimitSQL:
    """Regression: LIMIT must be pushed into SQL, not applied in Python after fetchall().

    Before fix: fetchall() pulled ALL rows, then Python sliced the result.
    After fix:  _effective_query() appends LIMIT N to the SQL before execution.
    """

    def test_effective_query_without_limit(self):
        loader = SnowflakeLoader(
            account="a", user="u", password="p", query="SELECT * FROM t"
        )
        assert loader._effective_query() == "SELECT * FROM t"

    def test_effective_query_appends_limit(self):
        loader = SnowflakeLoader(
            account="a", user="u", password="p",
            query="SELECT * FROM t", limit=10
        )
        q = loader._effective_query()
        assert q.endswith("LIMIT 10"), f"Expected LIMIT 10 in query, got: {q!r}"

    def test_effective_query_strips_trailing_semicolon_before_limit(self):
        loader = SnowflakeLoader(
            account="a", user="u", password="p",
            query="SELECT * FROM t;", limit=5
        )
        q = loader._effective_query()
        assert ";" not in q, "Semicolon should be stripped before LIMIT"
        assert q.endswith("LIMIT 5")

    def test_effective_query_strips_trailing_whitespace_before_limit(self):
        loader = SnowflakeLoader(
            account="a", user="u", password="p",
            query="SELECT * FROM t   ", limit=3
        )
        q = loader._effective_query()
        assert q.endswith("LIMIT 3")

    def test_limit_is_applied_in_sql_not_python(self):
        """Verify execute() receives the LIMIT-bearing query, not the plain one."""
        loader = SnowflakeLoader(
            account="a", user="u", password="p",
            query="SELECT * FROM t", limit=2
        )
        mock_pkg, mock_connector, _mock_conn, mock_cur = _make_snowflake_mock(
            rows=[("r1",), ("r2",)], columns=["col"]
        )
        with patch.dict(sys.modules, {"snowflake": mock_pkg, "snowflake.connector": mock_connector}):
            docs = loader.load()

        executed_query = mock_cur.execute.call_args[0][0]
        assert "LIMIT 2" in executed_query, (
            f"LIMIT must be in SQL, not Python: executed_query={executed_query!r}"
        )
        assert len(docs) == 2

    def test_no_limit_fetches_all_rows(self):
        loader = SnowflakeLoader(
            account="a", user="u", password="p",
            query="SELECT * FROM t"
        )
        mock_pkg, mock_connector, _mock_conn, mock_cur = _make_snowflake_mock(
            rows=[("v1",), ("v2",), ("v3",)], columns=["col"]
        )
        with patch.dict(sys.modules, {"snowflake": mock_pkg, "snowflake.connector": mock_connector}):
            docs = loader.load()

        executed_query = mock_cur.execute.call_args[0][0]
        assert "LIMIT" not in executed_query
        assert len(docs) == 3

    def test_limit_integer_cast_safety(self):
        """Ensure the limit value is always cast to int."""
        loader = SnowflakeLoader(
            account="a", user="u", password="p",
            query="SELECT 1", limit=50
        )
        q = loader._effective_query()
        assert q.endswith("LIMIT 50")
        assert ";" not in q


# ── document construction ─────────────────────────────────────────────────────


class TestSnowflakeLoaderDocuments:
    def _load_with_rows(self, rows, columns, text_fields=None, limit=None):
        loader = SnowflakeLoader(
            account="a", user="u", password="p",
            query="SELECT 1",
            text_fields=text_fields,
            limit=limit,
        )
        mock_pkg, mock_connector, _, _ = _make_snowflake_mock(rows=rows, columns=columns)
        with patch.dict(sys.modules, {"snowflake": mock_pkg, "snowflake.connector": mock_connector}):
            return loader.load()

    def test_all_columns_joined_as_text(self):
        docs = self._load_with_rows(rows=[("hello", "world")], columns=["a", "b"])
        assert len(docs) == 1
        assert "hello" in docs[0].text
        assert "world" in docs[0].text

    def test_text_fields_filter(self):
        docs = self._load_with_rows(
            rows=[("hello", "secret")], columns=["a", "b"], text_fields=["a"]
        )
        assert "hello" in docs[0].text
        assert "secret" not in docs[0].text

    def test_empty_row_skipped(self):
        docs = self._load_with_rows(rows=[(None, ""), ("real", "data")], columns=["a", "b"])
        assert len(docs) == 1
        assert "real" in docs[0].text

    def test_metadata_contains_source(self):
        docs = self._load_with_rows(rows=[("v",)], columns=["col"])
        assert docs[0].metadata["source"] == "snowflake"

    def test_import_error_raised_cleanly(self):
        loader = SnowflakeLoader(account="a", user="u", password="p", query="SELECT 1")
        # Remove snowflake from sys.modules so the ImportError path is exercised
        with patch.dict(sys.modules, {"snowflake": None, "snowflake.connector": None}):
            with pytest.raises(ImportError, match="snowflake"):
                loader.load()

    def test_query_error_raises_runtime_error(self):
        loader = SnowflakeLoader(account="a", user="u", password="p", query="BAD SQL")
        mock_pkg, mock_connector, _mock_conn, mock_cur = _make_snowflake_mock(rows=[], columns=[])
        mock_cur.execute.side_effect = RuntimeError("SQL compilation error")
        with patch.dict(sys.modules, {"snowflake": mock_pkg, "snowflake.connector": mock_connector}):
            with pytest.raises(RuntimeError, match="Snowflake query failed"):
                loader.load()


# ── async wrapper ─────────────────────────────────────────────────────────────


class TestSnowflakeLoaderAsync:
    @pytest.mark.asyncio
    async def test_aload_returns_documents(self):
        loader = SnowflakeLoader(account="a", user="u", password="p", query="SELECT 1")
        mock_pkg, mock_connector, _, _ = _make_snowflake_mock(
            rows=[("hello",)], columns=["col"]
        )
        with patch.dict(sys.modules, {"snowflake": mock_pkg, "snowflake.connector": mock_connector}):
            docs = await loader.aload()

        assert len(docs) == 1
        assert "hello" in docs[0].text
