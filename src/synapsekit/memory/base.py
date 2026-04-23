from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

MemoryType = Literal["episodic", "semantic"]


@dataclass(slots=True)
class MemoryRecord:
    id: str
    agent_id: str
    content: str
    memory_type: MemoryType
    embedding: list[float]
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    ttl_days: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self, now: datetime | None = None) -> bool:
        if self.ttl_days is None:
            return False
        now_utc = now or datetime.now(timezone.utc)
        return self.created_at + timedelta(days=self.ttl_days) <= now_utc


class BaseMemoryBackend(ABC):
    @abstractmethod
    async def store(self, record: MemoryRecord) -> None: ...

    @abstractmethod
    async def fetch(
        self,
        agent_id: str,
        memory_type: MemoryType | None = None,
        *,
        include_expired: bool = False,
    ) -> list[MemoryRecord]: ...

    @abstractmethod
    async def touch(
        self,
        agent_id: str,
        record_id: str,
        *,
        accessed_at: datetime | None = None,
    ) -> None: ...

    @abstractmethod
    async def delete(self, agent_id: str, record_id: str) -> bool: ...

    @abstractmethod
    async def clear(self, agent_id: str, memory_type: MemoryType | None = None) -> int: ...

    @abstractmethod
    async def count(self, agent_id: str, memory_type: MemoryType | None = None) -> int: ...

    @abstractmethod
    async def prune_expired(self, *, now: datetime | None = None) -> int: ...
