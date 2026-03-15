from __future__ import annotations

import sqlite3
from typing import Any

from ..base import BaseTool, ToolResult


class SQLSchemaInspectionTool(BaseTool):
    """Inspect database schema: list tables and describe columns."""

    name = "sql_schema"
    description = (
        "Inspect the schema of a SQL database. "
        "Actions: 'list_tables' returns all table names, "
        "'describe_table' returns column details for a given table. "
        "Input: action (required), table_name (optional, for describe_table)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "The action to perform: 'list_tables' or 'describe_table'",
                "enum": ["list_tables", "describe_table"],
            },
            "table_name": {
                "type": "string",
                "description": "Table name (required for describe_table)",
            },
        },
        "required": ["action"],
    }

    def __init__(self, connection_string: str) -> None:
        self._connection_string = connection_string

    def _is_sqlite(self) -> bool:
        return self._connection_string.endswith(".db") or self._connection_string == ":memory:"

    def _list_tables_sqlite(self) -> list[str]:
        conn = sqlite3.connect(self._connection_string)
        try:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            return [r[0] for r in rows]
        finally:
            conn.close()

    def _describe_table_sqlite(self, table_name: str) -> list[dict[str, Any]]:
        conn = sqlite3.connect(self._connection_string)
        try:
            rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            return [
                {
                    "name": r[1],
                    "type": r[2],
                    "nullable": not r[3],
                    "pk": bool(r[5]),
                }
                for r in rows
            ]
        finally:
            conn.close()

    def _list_tables_sqlalchemy(self) -> list[str]:
        try:
            from sqlalchemy import create_engine, inspect
        except ImportError:
            raise ImportError(
                "sqlalchemy required for non-SQLite databases: pip install sqlalchemy"
            ) from None
        engine = create_engine(self._connection_string)
        inspector = inspect(engine)
        return sorted(inspector.get_table_names())

    def _describe_table_sqlalchemy(self, table_name: str) -> list[dict[str, Any]]:
        try:
            from sqlalchemy import create_engine, inspect
        except ImportError:
            raise ImportError(
                "sqlalchemy required for non-SQLite databases: pip install sqlalchemy"
            ) from None
        engine = create_engine(self._connection_string)
        inspector = inspect(engine)
        columns = inspector.get_columns(table_name)
        pk_cols = {
            c for c in (inspector.get_pk_constraint(table_name).get("constrained_columns") or [])
        }
        return [
            {
                "name": col["name"],
                "type": str(col["type"]),
                "nullable": col.get("nullable", True),
                "pk": col["name"] in pk_cols,
            }
            for col in columns
        ]

    async def run(self, action: str = "", table_name: str = "", **kwargs: Any) -> ToolResult:
        if not action:
            return ToolResult(output="", error="action is required.")

        try:
            if action == "list_tables":
                if self._is_sqlite():
                    tables = self._list_tables_sqlite()
                else:
                    tables = self._list_tables_sqlalchemy()
                return ToolResult(output=", ".join(tables) if tables else "(no tables)")

            if action == "describe_table":
                if not table_name:
                    return ToolResult(output="", error="table_name is required for describe_table.")
                if self._is_sqlite():
                    cols = self._describe_table_sqlite(table_name)
                else:
                    cols = self._describe_table_sqlalchemy(table_name)
                if not cols:
                    return ToolResult(
                        output="", error=f"Table {table_name!r} not found or has no columns."
                    )
                lines = [
                    f"{c['name']} ({c['type']}, nullable={c['nullable']}, pk={c['pk']})"
                    for c in cols
                ]
                return ToolResult(output="\n".join(lines))

            return ToolResult(output="", error=f"Unknown action: {action!r}")
        except Exception as e:
            return ToolResult(output="", error=f"SQL schema error: {e}")
