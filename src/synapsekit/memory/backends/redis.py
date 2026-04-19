from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from ..base import BaseMemoryBackend, MemoryRecord, MemoryType


class RedisMemoryBackend(BaseMemoryBackend):
    """Redis-backed persistent memory backend."""

    def __init__(
        self, url: str = "redis://localhost:6379", prefix: str = "synapsekit:agentmem:"
    ) -> None:
        try:
            import redis.asyncio as redis
        except ImportError:
            raise ImportError("redis package required: pip install synapsekit[redis]") from None

        self._redis = redis
        self._client = redis.from_url(url, decode_responses=True)
        self._prefix = prefix

    def _record_key(self, agent_id: str, record_id: str) -> str:
        return f"{self._prefix}{agent_id}:record:{record_id}"

    def _index_key(self, agent_id: str) -> str:
        return f"{self._prefix}{agent_id}:index"

    @staticmethod
    def _serialize(record: MemoryRecord) -> str:
        payload = {
            "id": record.id,
            "agent_id": record.agent_id,
            "content": record.content,
            "memory_type": record.memory_type,
            "embedding": record.embedding,
            "created_at": record.created_at.timestamp(),
            "accessed_at": record.accessed_at.timestamp(),
            "access_count": record.access_count,
            "ttl_days": record.ttl_days,
            "metadata": record.metadata,
        }
        return json.dumps(payload)

    @staticmethod
    def _deserialize(raw: str) -> MemoryRecord:
        item: dict[str, Any] = json.loads(raw)
        return MemoryRecord(
            id=str(item["id"]),
            agent_id=str(item["agent_id"]),
            content=str(item["content"]),
            memory_type=item["memory_type"],
            embedding=list(item["embedding"]),
            created_at=datetime.fromtimestamp(float(item["created_at"]), tz=timezone.utc),
            accessed_at=datetime.fromtimestamp(float(item["accessed_at"]), tz=timezone.utc),
            access_count=int(item.get("access_count", 0)),
            ttl_days=item.get("ttl_days"),
            metadata=dict(item.get("metadata") or {}),
        )

    async def store(self, record: MemoryRecord) -> None:
        key = self._record_key(record.agent_id, record.id)
        index = self._index_key(record.agent_id)
        pipe = self._client.pipeline()
        pipe.set(key, self._serialize(record))
        pipe.zadd(index, {record.id: record.created_at.timestamp()})
        await pipe.execute()

    async def fetch(
        self,
        agent_id: str,
        memory_type: MemoryType | None = None,
        *,
        include_expired: bool = False,
    ) -> list[MemoryRecord]:
        ids: list[str] = await self._client.zrange(self._index_key(agent_id), 0, -1)
        if not ids:
            return []

        keys = [self._record_key(agent_id, rid) for rid in ids]
        raw_records = await self._client.mget(keys)
        now = datetime.now(timezone.utc)

        out: list[MemoryRecord] = []
        for raw in raw_records:
            if not raw:
                continue
            rec = self._deserialize(raw)
            if memory_type is not None and rec.memory_type != memory_type:
                continue
            if not include_expired and rec.is_expired(now):
                continue
            out.append(rec)
        out.sort(key=lambda r: r.created_at)
        return out

    async def touch(
        self,
        agent_id: str,
        record_id: str,
        *,
        accessed_at: datetime | None = None,
    ) -> None:
        key = self._record_key(agent_id, record_id)
        raw = await self._client.get(key)
        if not raw:
            return
        rec = self._deserialize(raw)
        rec.accessed_at = accessed_at or datetime.now(timezone.utc)
        rec.access_count += 1
        await self._client.set(key, self._serialize(rec))

    async def delete(self, agent_id: str, record_id: str) -> bool:
        key = self._record_key(agent_id, record_id)
        index = self._index_key(agent_id)
        pipe = self._client.pipeline()
        pipe.delete(key)
        pipe.zrem(index, record_id)
        deleted, _ = await pipe.execute()
        return int(deleted) > 0

    async def clear(self, agent_id: str, memory_type: MemoryType | None = None) -> int:
        if memory_type is None:
            ids: list[str] = await self._client.zrange(self._index_key(agent_id), 0, -1)
            if not ids:
                return 0
            keys = [self._record_key(agent_id, rid) for rid in ids]
            pipe = self._client.pipeline()
            pipe.delete(*keys)
            pipe.delete(self._index_key(agent_id))
            await pipe.execute()
            return len(ids)

        records = await self.fetch(agent_id, memory_type=memory_type, include_expired=True)
        if not records:
            return 0
        pipe = self._client.pipeline()
        for rec in records:
            pipe.delete(self._record_key(agent_id, rec.id))
            pipe.zrem(self._index_key(agent_id), rec.id)
        await pipe.execute()
        return len(records)

    async def count(self, agent_id: str, memory_type: MemoryType | None = None) -> int:
        if memory_type is None:
            return int(await self._client.zcard(self._index_key(agent_id)))
        records = await self.fetch(agent_id, memory_type=memory_type)
        return len(records)

    async def prune_expired(self, *, now: datetime | None = None) -> int:
        now_utc = now or datetime.now(timezone.utc)
        removed = 0

        cursor = 0
        pattern = f"{self._prefix}*:index"
        while True:
            cursor, keys = await self._client.scan(cursor=cursor, match=pattern, count=100)
            for index_key in keys:
                agent_id = index_key.split(":")[-2]
                records = await self.fetch(agent_id, include_expired=True)
                for rec in records:
                    if rec.is_expired(now_utc) and await self.delete(agent_id, rec.id):
                        removed += 1
            if cursor == 0:
                break
        return removed

    async def close(self) -> None:
        await self._client.aclose()
