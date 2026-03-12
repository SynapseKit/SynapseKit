from __future__ import annotations

import sqlite3
from typing import Any

from ._cache import AsyncLRUCache


class SQLiteLLMCache:
    """Persistent LLM cache backed by SQLite.

    Stores cache entries on disk so they survive process restarts.
    Uses the same ``make_key`` logic as :class:`AsyncLRUCache`.

    Usage::

        from synapsekit.llm._sqlite_cache import SQLiteLLMCache

        cache = SQLiteLLMCache("llm_cache.db")
        cache.put(key, value)
        cached = cache.get(key)
    """

    make_key = staticmethod(AsyncLRUCache.make_key)

    def __init__(self, db_path: str = "synapsekit_llm_cache.db") -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS llm_cache (key TEXT PRIMARY KEY, value TEXT)"
        )
        self._conn.commit()
        self.hits: int = 0
        self.misses: int = 0

    def get(self, key: str) -> Any | None:
        row = self._conn.execute("SELECT value FROM llm_cache WHERE key = ?", (key,)).fetchone()
        if row is not None:
            self.hits += 1
            return row[0]
        self.misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO llm_cache (key, value) VALUES (?, ?)",
            (key, str(value)),
        )
        self._conn.commit()

    def clear(self) -> None:
        self._conn.execute("DELETE FROM llm_cache")
        self._conn.commit()

    def __len__(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        self._conn.close()
