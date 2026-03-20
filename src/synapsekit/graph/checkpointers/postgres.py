"""PostgreSQL-backed graph checkpointer."""

from __future__ import annotations

import json
from typing import Any

from .base import BaseCheckpointer

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS synapsekit_checkpoints (
    graph_id TEXT PRIMARY KEY,
    step INTEGER NOT NULL,
    state JSONB NOT NULL
)
"""

_UPSERT_SQL = """
INSERT INTO synapsekit_checkpoints (graph_id, step, state)
VALUES (%s, %s, %s)
ON CONFLICT (graph_id) DO UPDATE SET step = EXCLUDED.step, state = EXCLUDED.state
"""

_SELECT_SQL = "SELECT step, state FROM synapsekit_checkpoints WHERE graph_id = %s"
_DELETE_SQL = "DELETE FROM synapsekit_checkpoints WHERE graph_id = %s"


class PostgresCheckpointer(BaseCheckpointer):
    """Persist graph checkpoints in PostgreSQL.

    Requires ``psycopg`` (``pip install synapsekit[postgres]``).

    Usage::

        import psycopg
        conn = psycopg.connect("postgresql://localhost/mydb")
        cp = PostgresCheckpointer(conn)
        cp.save("my-graph", 3, {"messages": [...]})
        step, state = cp.load("my-graph")
    """

    def __init__(self, connection: Any, *, autocommit: bool = True) -> None:
        """
        Args:
            connection: A ``psycopg.Connection`` instance.
            autocommit: Whether to commit after each operation.
        """
        self._conn = connection
        self._autocommit = autocommit
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create the checkpoints table if it doesn't exist."""
        with self._conn.cursor() as cur:
            cur.execute(_CREATE_TABLE_SQL)
        if self._autocommit:
            self._conn.commit()

    def save(self, graph_id: str, step: int, state: dict[str, Any]) -> None:
        """Persist the state at the given step using UPSERT."""
        state_json = json.dumps(state)
        with self._conn.cursor() as cur:
            cur.execute(_UPSERT_SQL, (graph_id, step, state_json))
        if self._autocommit:
            self._conn.commit()

    def load(self, graph_id: str) -> tuple[int, dict[str, Any]] | None:
        """Load the most recent checkpoint. Returns ``(step, state)`` or ``None``."""
        with self._conn.cursor() as cur:
            cur.execute(_SELECT_SQL, (graph_id,))
            row = cur.fetchone()
        if row is None:
            return None
        step, state = row
        # psycopg auto-deserializes JSONB, but handle string case too
        if isinstance(state, str):
            state = json.loads(state)
        return step, state

    def delete(self, graph_id: str) -> None:
        """Remove the checkpoint for the given graph_id."""
        with self._conn.cursor() as cur:
            cur.execute(_DELETE_SQL, (graph_id,))
        if self._autocommit:
            self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
