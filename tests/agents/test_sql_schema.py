from __future__ import annotations

import sqlite3

import pytest

from synapsekit.agents.tools.sql_schema import SQLSchemaInspectionTool


@pytest.fixture
def sqlite_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT)")
    conn.execute("CREATE TABLE posts (id INTEGER PRIMARY KEY, user_id INTEGER, title TEXT)")
    conn.commit()
    conn.close()
    return db_path


class TestSQLSchemaInspectionTool:
    @pytest.mark.asyncio
    async def test_list_tables(self, sqlite_db):
        tool = SQLSchemaInspectionTool(connection_string=sqlite_db)
        result = await tool.run(action="list_tables")
        assert not result.is_error
        assert "users" in result.output
        assert "posts" in result.output

    @pytest.mark.asyncio
    async def test_describe_table(self, sqlite_db):
        tool = SQLSchemaInspectionTool(connection_string=sqlite_db)
        result = await tool.run(action="describe_table", table_name="users")
        assert not result.is_error
        assert "id" in result.output
        assert "name" in result.output
        assert "email" in result.output
        assert "pk=True" in result.output

    @pytest.mark.asyncio
    async def test_describe_missing_table_name(self, sqlite_db):
        tool = SQLSchemaInspectionTool(connection_string=sqlite_db)
        result = await tool.run(action="describe_table")
        assert result.is_error
        assert "table_name is required" in result.error

    @pytest.mark.asyncio
    async def test_no_action(self, sqlite_db):
        tool = SQLSchemaInspectionTool(connection_string=sqlite_db)
        result = await tool.run()
        assert result.is_error
        assert "action is required" in result.error

    @pytest.mark.asyncio
    async def test_unknown_action(self, sqlite_db):
        tool = SQLSchemaInspectionTool(connection_string=sqlite_db)
        result = await tool.run(action="drop_table")
        assert result.is_error
        assert "Unknown action" in result.error

    def test_schema(self, sqlite_db):
        tool = SQLSchemaInspectionTool(connection_string=sqlite_db)
        schema = tool.schema()
        assert schema["function"]["name"] == "sql_schema"
