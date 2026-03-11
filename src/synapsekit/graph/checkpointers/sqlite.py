from __future__ import annotations

import json
import sqlite3
from typing import Any

from .base import BaseCheckpointer


class SQLiteCheckpointer(BaseCheckpointer):
    """SQLite-backed checkpointer using stdlib ``sqlite3``."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS checkpoints "
            "(graph_id TEXT PRIMARY KEY, step INTEGER, state TEXT)"
        )
        self._conn.commit()

    def save(self, graph_id: str, step: int, state: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO checkpoints (graph_id, step, state) VALUES (?, ?, ?)",
            (graph_id, step, json.dumps(state)),
        )
        self._conn.commit()

    def load(self, graph_id: str) -> tuple[int, dict[str, Any]] | None:
        row = self._conn.execute(
            "SELECT step, state FROM checkpoints WHERE graph_id = ?", (graph_id,)
        ).fetchone()
        if row is None:
            return None
        return row[0], json.loads(row[1])

    def delete(self, graph_id: str) -> None:
        self._conn.execute("DELETE FROM checkpoints WHERE graph_id = ?", (graph_id,))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
