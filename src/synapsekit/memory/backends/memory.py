from __future__ import annotations

from datetime import datetime, timezone

from ..base import BaseMemoryBackend, MemoryRecord, MemoryType


class InMemoryMemoryBackend(BaseMemoryBackend):
    """In-process backend for tests/dev."""

    def __init__(self) -> None:
        self._records: dict[str, dict[str, MemoryRecord]] = {}

    async def store(self, record: MemoryRecord) -> None:
        self._records.setdefault(record.agent_id, {})[record.id] = record

    async def fetch(
        self,
        agent_id: str,
        memory_type: MemoryType | None = None,
        *,
        include_expired: bool = False,
    ) -> list[MemoryRecord]:
        bucket = self._records.get(agent_id, {})
        now = datetime.now(timezone.utc)
        out: list[MemoryRecord] = []
        for rec in bucket.values():
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
        rec = self._records.get(agent_id, {}).get(record_id)
        if rec is None:
            return
        rec.accessed_at = accessed_at or datetime.now(timezone.utc)
        rec.access_count += 1

    async def delete(self, agent_id: str, record_id: str) -> bool:
        bucket = self._records.get(agent_id)
        if bucket is None:
            return False
        return bucket.pop(record_id, None) is not None

    async def clear(self, agent_id: str, memory_type: MemoryType | None = None) -> int:
        bucket = self._records.get(agent_id)
        if bucket is None:
            return 0
        if memory_type is None:
            removed = len(bucket)
            bucket.clear()
            return removed

        ids = [rid for rid, rec in bucket.items() if rec.memory_type == memory_type]
        for rid in ids:
            bucket.pop(rid, None)
        return len(ids)

    async def count(self, agent_id: str, memory_type: MemoryType | None = None) -> int:
        bucket = self._records.get(agent_id, {})
        if memory_type is None:
            return len(bucket)
        return sum(1 for rec in bucket.values() if rec.memory_type == memory_type)

    async def prune_expired(self, *, now: datetime | None = None) -> int:
        now_utc = now or datetime.now(timezone.utc)
        removed = 0
        for agent_id, bucket in list(self._records.items()):
            to_remove = [rid for rid, rec in bucket.items() if rec.is_expired(now_utc)]
            for rid in to_remove:
                bucket.pop(rid, None)
                removed += 1
            if not bucket:
                self._records.pop(agent_id, None)
        return removed
