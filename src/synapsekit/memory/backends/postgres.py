from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from ..base import BaseMemoryBackend, MemoryRecord, MemoryType


class PostgresMemoryBackend(BaseMemoryBackend):
    """PostgreSQL-backed persistent memory backend via asyncpg."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: Any | None = None

    async def _ensure_pool(self):
        if self._pool is not None:
            return self._pool
        try:
            import asyncpg
        except ImportError:
            raise ImportError("asyncpg required: pip install asyncpg") from None

        self._pool = await asyncpg.create_pool(self._dsn)
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_memory_records (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    embedding JSONB NOT NULL,
                    created_at DOUBLE PRECISION NOT NULL,
                    accessed_at DOUBLE PRECISION NOT NULL,
                    access_count INTEGER NOT NULL,
                    ttl_days INTEGER,
                    metadata JSONB NOT NULL
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pg_memory_agent_type_created "
                "ON agent_memory_records(agent_id, memory_type, created_at)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pg_memory_agent_created "
                "ON agent_memory_records(agent_id, created_at)"
            )
        return self._pool

    @staticmethod
    def _to_record(row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            agent_id=row["agent_id"],
            content=row["content"],
            memory_type=row["memory_type"],
            embedding=list(row["embedding"]),
            created_at=datetime.fromtimestamp(float(row["created_at"]), tz=timezone.utc),
            accessed_at=datetime.fromtimestamp(float(row["accessed_at"]), tz=timezone.utc),
            access_count=int(row["access_count"]),
            ttl_days=row["ttl_days"],
            metadata=dict(row["metadata"]),
        )

    async def store(self, record: MemoryRecord) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agent_memory_records (
                    id, agent_id, content, memory_type, embedding,
                    created_at, accessed_at, access_count, ttl_days, metadata
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
                ON CONFLICT (id) DO UPDATE
                    SET content = EXCLUDED.content,
                        memory_type = EXCLUDED.memory_type,
                        embedding = EXCLUDED.embedding,
                        created_at = EXCLUDED.created_at,
                        accessed_at = EXCLUDED.accessed_at,
                        access_count = EXCLUDED.access_count,
                        ttl_days = EXCLUDED.ttl_days,
                        metadata = EXCLUDED.metadata
                """,
                record.id,
                record.agent_id,
                record.content,
                record.memory_type,
                json.dumps(record.embedding),
                record.created_at.timestamp(),
                record.accessed_at.timestamp(),
                record.access_count,
                record.ttl_days,
                json.dumps(record.metadata),
            )

    async def fetch(
        self,
        agent_id: str,
        memory_type: MemoryType | None = None,
        *,
        include_expired: bool = False,
    ) -> list[MemoryRecord]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            if memory_type is None:
                rows = await conn.fetch(
                    "SELECT * FROM agent_memory_records WHERE agent_id = $1 ORDER BY created_at ASC",
                    agent_id,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT * FROM agent_memory_records
                    WHERE agent_id = $1 AND memory_type = $2
                    ORDER BY created_at ASC
                    """,
                    agent_id,
                    memory_type,
                )

        records = [self._to_record(r) for r in rows]
        if include_expired:
            return records
        now = datetime.now(timezone.utc)
        return [r for r in records if not r.is_expired(now)]

    async def touch(
        self,
        agent_id: str,
        record_id: str,
        *,
        accessed_at: datetime | None = None,
    ) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE agent_memory_records
                SET accessed_at = $1, access_count = access_count + 1
                WHERE agent_id = $2 AND id = $3
                """,
                (accessed_at or datetime.now(timezone.utc)).timestamp(),
                agent_id,
                record_id,
            )

    async def delete(self, agent_id: str, record_id: str) -> bool:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM agent_memory_records WHERE agent_id = $1 AND id = $2",
                agent_id,
                record_id,
            )
        return str(result).endswith("1")

    async def clear(self, agent_id: str, memory_type: MemoryType | None = None) -> int:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            if memory_type is None:
                result = await conn.execute(
                    "DELETE FROM agent_memory_records WHERE agent_id = $1",
                    agent_id,
                )
            else:
                result = await conn.execute(
                    "DELETE FROM agent_memory_records WHERE agent_id = $1 AND memory_type = $2",
                    agent_id,
                    memory_type,
                )
        return int(result.split()[-1])

    async def count(self, agent_id: str, memory_type: MemoryType | None = None) -> int:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            if memory_type is None:
                row = await conn.fetchrow(
                    "SELECT COUNT(*) AS n FROM agent_memory_records WHERE agent_id = $1",
                    agent_id,
                )
            else:
                row = await conn.fetchrow(
                    """
                    SELECT COUNT(*) AS n FROM agent_memory_records
                    WHERE agent_id = $1 AND memory_type = $2
                    """,
                    agent_id,
                    memory_type,
                )
        return int(row["n"]) if row else 0

    async def prune_expired(self, *, now: datetime | None = None) -> int:
        ts = (now or datetime.now(timezone.utc)).timestamp()
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM agent_memory_records
                WHERE ttl_days IS NOT NULL
                  AND (created_at + (ttl_days * 86400.0)) <= $1
                """,
                ts,
            )
        return int(result.split()[-1])

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
