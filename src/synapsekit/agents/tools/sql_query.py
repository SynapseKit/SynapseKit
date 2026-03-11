from __future__ import annotations

from typing import Any, Optional

from ..base import BaseTool, ToolResult


class SQLQueryTool(BaseTool):
    """Execute a SQL query and return results as text."""

    name = "sql_query"
    description = (
        "Execute a SQL SELECT query against a database and return results. "
        "Input: a SQL SELECT query string."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A SQL SELECT query to execute",
            }
        },
        "required": ["query"],
    }

    def __init__(
        self,
        connection_string: str,
        max_rows: int = 50,
    ) -> None:
        """
        Args:
            connection_string: SQLite path (e.g. "mydb.sqlite") or SQLAlchemy URL
                               (e.g. "postgresql://user:pass@host/db").
            max_rows: Maximum rows to return per query.
        """
        self._connection_string = connection_string
        self._max_rows = max_rows
        self._is_sqlite = not connection_string.startswith(
            ("postgresql", "mysql", "oracle", "mssql", "sqlite+")
        )

    def _get_connection(self):
        if self._is_sqlite:
            import sqlite3
            return sqlite3.connect(self._connection_string)
        try:
            from sqlalchemy import create_engine, text as sa_text
            engine = create_engine(self._connection_string)
            return engine.connect(), sa_text, True
        except ImportError:
            raise ImportError(
                "sqlalchemy required for non-SQLite databases: pip install sqlalchemy"
            )

    async def run(self, query: str = "", **kwargs: Any) -> ToolResult:
        sql = query or kwargs.get("input", "")
        if not sql:
            return ToolResult(output="", error="No SQL query provided.")

        # Safety: only allow SELECT
        stripped = sql.strip().upper()
        if not stripped.startswith("SELECT"):
            return ToolResult(
                output="",
                error="Only SELECT queries are allowed for safety.",
            )

        try:
            if self._is_sqlite:
                import sqlite3
                conn = sqlite3.connect(self._connection_string)
                try:
                    cursor = conn.cursor()
                    cursor.execute(sql)
                    cols = [d[0] for d in cursor.description] if cursor.description else []
                    rows = cursor.fetchmany(self._max_rows)
                finally:
                    conn.close()
            else:
                from sqlalchemy import create_engine, text as sa_text
                engine = create_engine(self._connection_string)
                with engine.connect() as conn:
                    result = conn.execute(sa_text(sql))
                    cols = list(result.keys())
                    rows = result.fetchmany(self._max_rows)

            if not rows:
                return ToolResult(output="Query returned no rows.")

            # Format as markdown table
            header = " | ".join(cols)
            separator = " | ".join(["---"] * len(cols))
            data_rows = [" | ".join(str(v) for v in row) for row in rows]
            table = "\n".join([header, separator] + data_rows)
            return ToolResult(output=table)

        except Exception as e:
            return ToolResult(output="", error=f"SQL error: {e}")
