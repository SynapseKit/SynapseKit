from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from ..base import BaseMemoryBackend, MemoryRecord, MemoryType


class SQLiteMemoryBackend(BaseMemoryBackend):
    """SQLite-backed persistent memory backend."""

    def __init__(self, path: str = "agent_memory.db") -> None:
        self._path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path, check_same_thread=False)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_memory_records (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    accessed_at REAL NOT NULL,
                    access_count INTEGER NOT NULL,
                    ttl_days INTEGER,
                    metadata TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_agent_type_created "
                "ON agent_memory_records(agent_id, memory_type, created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_agent_created "
                "ON agent_memory_records(agent_id, created_at)"
            )
            conn.commit()

    @staticmethod
    def _to_ts(dt: datetime) -> float:
        return dt.timestamp()

    @staticmethod
    def _from_row(row: tuple) -> MemoryRecord:
        (
            rec_id,
            agent_id,
            content,
            memory_type,
            embedding_json,
            created_ts,
            accessed_ts,
            access_count,
            ttl_days,
            metadata_json,
        ) = row
        return MemoryRecord(
            id=rec_id,
            agent_id=agent_id,
            content=content,
            memory_type=memory_type,
            embedding=list(json.loads(embedding_json)),
            created_at=datetime.fromtimestamp(created_ts, tz=timezone.utc),
            accessed_at=datetime.fromtimestamp(accessed_ts, tz=timezone.utc),
            access_count=int(access_count),
            ttl_days=None if ttl_days is None else int(ttl_days),
            metadata=dict(json.loads(metadata_json)),
        )

    async def store(self, record: MemoryRecord) -> None:
        def _op() -> None:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO agent_memory_records (
                        id, agent_id, content, memory_type, embedding,
                        created_at, accessed_at, access_count, ttl_days, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.id,
                        record.agent_id,
                        record.content,
                        record.memory_type,
                        json.dumps(record.embedding),
                        self._to_ts(record.created_at),
                        self._to_ts(record.accessed_at),
                        record.access_count,
                        record.ttl_days,
                        json.dumps(record.metadata),
                    ),
                )
                conn.commit()

        await asyncio.to_thread(_op)

    async def fetch(
        self,
        agent_id: str,
        memory_type: MemoryType | None = None,
        *,
        include_expired: bool = False,
    ) -> list[MemoryRecord]:
        def _op() -> list[MemoryRecord]:
            with self._connect() as conn:
                if memory_type is None:
                    rows = conn.execute(
                        """
                        SELECT id, agent_id, content, memory_type, embedding,
                               created_at, accessed_at, access_count, ttl_days, metadata
                        FROM agent_memory_records
                        WHERE agent_id = ?
                        ORDER BY created_at ASC
                        """,
                        (agent_id,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT id, agent_id, content, memory_type, embedding,
                               created_at, accessed_at, access_count, ttl_days, metadata
                        FROM agent_memory_records
                        WHERE agent_id = ? AND memory_type = ?
                        ORDER BY created_at ASC
                        """,
                        (agent_id, memory_type),
                    ).fetchall()

            records = [self._from_row(r) for r in rows]
            if include_expired:
                return records
            now = datetime.now(timezone.utc)
            return [r for r in records if not r.is_expired(now)]

        return await asyncio.to_thread(_op)

    async def touch(
        self,
        agent_id: str,
        record_id: str,
        *,
        accessed_at: datetime | None = None,
    ) -> None:
        ts = self._to_ts(accessed_at or datetime.now(timezone.utc))

        def _op() -> None:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE agent_memory_records
                    SET accessed_at = ?, access_count = access_count + 1
                    WHERE agent_id = ? AND id = ?
                    """,
                    (ts, agent_id, record_id),
                )
                conn.commit()

        await asyncio.to_thread(_op)

    async def delete(self, agent_id: str, record_id: str) -> bool:
        def _op() -> bool:
            with self._connect() as conn:
                cur = conn.execute(
                    "DELETE FROM agent_memory_records WHERE agent_id = ? AND id = ?",
                    (agent_id, record_id),
                )
                conn.commit()
                return cur.rowcount > 0

        return await asyncio.to_thread(_op)

    async def clear(self, agent_id: str, memory_type: MemoryType | None = None) -> int:
        def _op() -> int:
            with self._connect() as conn:
                if memory_type is None:
                    cur = conn.execute(
                        "DELETE FROM agent_memory_records WHERE agent_id = ?",
                        (agent_id,),
                    )
                else:
                    cur = conn.execute(
                        "DELETE FROM agent_memory_records WHERE agent_id = ? AND memory_type = ?",
                        (agent_id, memory_type),
                    )
                conn.commit()
                return int(cur.rowcount)

        return await asyncio.to_thread(_op)

    async def count(self, agent_id: str, memory_type: MemoryType | None = None) -> int:
        def _op() -> int:
            with self._connect() as conn:
                if memory_type is None:
                    row = conn.execute(
                        "SELECT COUNT(*) FROM agent_memory_records WHERE agent_id = ?",
                        (agent_id,),
                    ).fetchone()
                else:
                    row = conn.execute(
                        "SELECT COUNT(*) FROM agent_memory_records WHERE agent_id = ? AND memory_type = ?",
                        (agent_id, memory_type),
                    ).fetchone()
                return int(row[0]) if row else 0

        return await asyncio.to_thread(_op)

    async def prune_expired(self, *, now: datetime | None = None) -> int:
        now_ts = (now or datetime.now(timezone.utc)).timestamp()

        def _op() -> int:
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    DELETE FROM agent_memory_records
                    WHERE ttl_days IS NOT NULL
                      AND (created_at + (ttl_days * 86400.0)) <= ?
                    """,
                    (now_ts,),
                )
                conn.commit()
                return int(cur.rowcount)

        return await asyncio.to_thread(_op)
