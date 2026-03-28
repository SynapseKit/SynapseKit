import sqlite3

import pytest

from synapsekit.agents.tools.sql_query import SQLQueryTool


@pytest.fixture
def test_db(tmp_path):
    db_path = tmp_path / "test.sqlite"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
    cursor.executemany(
        "INSERT INTO users (name, age) VALUES (?, ?)",
        [("Alice", 30), ("Bob", 25), ("Charlie", 35)],
    )
    conn.commit()
    conn.close()
    return str(db_path)


@pytest.mark.asyncio
async def test_sql_query_select(test_db):
    tool = SQLQueryTool(connection_string=test_db)
    result = await tool.run(query="SELECT * FROM users ORDER BY id")
    assert result.error is None

    # Check table formatting
    assert "id | name | age" in result.output
    assert "--- | --- | ---" in result.output
    assert "1 | Alice | 30" in result.output
    assert "2 | Bob | 25" in result.output


@pytest.mark.asyncio
async def test_sql_query_parameterized(test_db):
    tool = SQLQueryTool(connection_string=test_db)
    # Using named parameters for sqlite3 parameterized query testing
    result = await tool.run(
        query="SELECT * FROM users WHERE age > :min_age", params={"min_age": 28}
    )
    assert result.error is None

    assert "Alice" in result.output
    assert "Charlie" in result.output
    assert "Bob" not in result.output


@pytest.mark.asyncio
async def test_sql_query_security_only_select(test_db):
    tool = SQLQueryTool(connection_string=test_db)
    result = await tool.run(query="DROP TABLE users;")

    assert result.error is not None
    assert "Only SELECT queries are allowed" in result.error

    # Verify table still exists
    conn = sqlite3.connect(test_db)
    cursor = conn.cursor()
    cursor.execute("SELECT count(*) FROM users")
    count = cursor.fetchone()[0]
    conn.close()
    assert count == 3


@pytest.mark.asyncio
async def test_sql_query_empty_result(test_db):
    tool = SQLQueryTool(connection_string=test_db)
    result = await tool.run(query="SELECT * FROM users WHERE age > 100")
    assert result.error is None
    assert "Query returned no rows" in result.output


@pytest.mark.asyncio
async def test_sql_query_max_rows(test_db):
    tool = SQLQueryTool(connection_string=test_db, max_rows=2)
    result = await tool.run(query="SELECT * FROM users ORDER BY id")

    assert "1 | Alice | 30" in result.output
    assert "2 | Bob | 25" in result.output
    assert "3 | Charlie | 35" not in result.output


@pytest.mark.asyncio
async def test_sql_query_invalid_query(test_db):
    tool = SQLQueryTool(connection_string=test_db)
    result = await tool.run(query="SELECT * FROM table_that_does_not_exist")

    assert result.error is not None
    assert "SQL error" in result.error
    assert "no such table" in result.error
