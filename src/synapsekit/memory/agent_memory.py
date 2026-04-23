from __future__ import annotations

import hashlib
import inspect
import math
import re
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from .backends import (
    InMemoryMemoryBackend,
    PostgresMemoryBackend,
    RedisMemoryBackend,
    SQLiteMemoryBackend,
)
from .base import BaseMemoryBackend, MemoryRecord, MemoryType

EmbedderFn = Callable[[str], list[float] | Awaitable[list[float]]]


class AgentMemory:
    """Persistent agent memory orchestrator with episodic + semantic tiers."""

    def __init__(
        self,
        backend: str | BaseMemoryBackend = "memory",
        *,
        path: str | None = None,
        redis_url: str = "redis://localhost:6379",
        postgres_dsn: str | None = None,
        embedder: EmbedderFn | None = None,
        llm: Any | None = None,
        max_episodes: int = 100,
        consolidation_window: int = 20,
    ) -> None:
        self._backend = self._resolve_backend(
            backend,
            path=path,
            redis_url=redis_url,
            postgres_dsn=postgres_dsn,
        )
        self._embedder = embedder
        self._llm = llm
        self._max_episodes = max_episodes
        self._consolidation_window = consolidation_window

    @staticmethod
    def _resolve_backend(
        backend: str | BaseMemoryBackend,
        *,
        path: str | None,
        redis_url: str,
        postgres_dsn: str | None,
    ) -> BaseMemoryBackend:
        if isinstance(backend, BaseMemoryBackend):
            return backend
        if backend == "memory":
            return InMemoryMemoryBackend()
        if backend == "sqlite":
            return SQLiteMemoryBackend(path or "agent_memory.db")
        if backend == "redis":
            return RedisMemoryBackend(url=redis_url)
        if backend == "postgres":
            if not postgres_dsn:
                raise ValueError("postgres_dsn is required when backend='postgres'")
            return PostgresMemoryBackend(postgres_dsn)
        raise ValueError(f"Unknown backend: {backend!r}")

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _default_embed(text: str, dim: int = 128) -> list[float]:
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return [0.0] * dim
        vec = [0.0] * dim
        for token in tokens:
            h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h, byteorder="big") % dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec
        return [v / norm for v in vec]

    async def _embed_text(self, text: str) -> list[float]:
        if self._embedder is None:
            return self._default_embed(text)
        value = self._embedder(text)
        if inspect.isawaitable(value):
            value = await value
        return [float(x) for x in value]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    async def _store_record(
        self,
        *,
        agent_id: str,
        content: str,
        memory_type: MemoryType,
        ttl_days: int | None,
        metadata: dict[str, Any] | None,
        skip_auto_consolidate: bool,
    ) -> MemoryRecord:
        now = self._now()
        record = MemoryRecord(
            id=str(uuid.uuid4()),
            agent_id=agent_id,
            content=content,
            memory_type=memory_type,
            embedding=await self._embed_text(content),
            created_at=now,
            accessed_at=now,
            access_count=0,
            ttl_days=ttl_days,
            metadata=dict(metadata or {}),
        )
        await self._backend.store(record)

        if memory_type == "episodic" and not skip_auto_consolidate:
            episodic_count = await self._backend.count(agent_id, memory_type="episodic")
            if episodic_count > self._max_episodes:
                await self.consolidate(agent_id)
        return record

    async def store(
        self,
        *,
        agent_id: str,
        content: str,
        memory_type: MemoryType = "episodic",
        ttl_days: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        return await self._store_record(
            agent_id=agent_id,
            content=content,
            memory_type=memory_type,
            ttl_days=ttl_days,
            metadata=metadata,
            skip_auto_consolidate=False,
        )

    async def recall(
        self,
        *,
        agent_id: str,
        query: str,
        top_k: int = 5,
        memory_types: tuple[MemoryType, ...] = ("semantic", "episodic"),
    ) -> list[MemoryRecord]:
        if top_k <= 0:
            return []

        await self._backend.prune_expired()

        records: list[MemoryRecord] = []
        seen: set[str] = set()
        for memory_type in memory_types:
            batch = await self._backend.fetch(agent_id, memory_type=memory_type)
            for rec in batch:
                if rec.id in seen:
                    continue
                seen.add(rec.id)
                records.append(rec)

        if not records:
            return []

        q_emb = await self._embed_text(query)
        now = self._now()

        def _score(rec: MemoryRecord) -> float:
            sim = self._cosine(q_emb, rec.embedding)
            age_days = max((now - rec.created_at).total_seconds() / 86400.0, 0.0)
            recency = 1.0 / (1.0 + age_days)
            access = min(rec.access_count, 20) / 20.0
            semantic_boost = 0.05 if rec.memory_type == "semantic" else 0.0
            return (0.82 * sim) + (0.12 * recency) + (0.04 * access) + semantic_boost

        ranked = sorted(records, key=_score, reverse=True)[:top_k]

        for rec in ranked:
            await self._backend.touch(agent_id, rec.id)

        return ranked

    async def _summarize_contents(self, contents: list[str]) -> str:
        lines = [f"- {c.strip()}" for c in contents if c.strip()]
        if not lines:
            return ""

        if self._llm is not None:
            prompt = (
                "Consolidate these episodic memories into concise semantic facts. "
                "Return 3-8 bullet points with stable long-term information only.\n\n"
                + "\n".join(lines)
            )
            try:
                summary = await self._llm.generate(prompt)
                summary_text = str(summary).strip()
                if summary_text:
                    return summary_text
            except Exception:
                pass

        # Fallback deterministic summarization
        unique = list(dict.fromkeys(s[2:] for s in lines))
        if not unique:
            return ""
        return "; ".join(unique[:8])

    async def consolidate(
        self,
        agent_id: str,
        *,
        limit: int | None = None,
        delete_consolidated: bool | None = None,
    ) -> MemoryRecord | None:
        episodic = await self._backend.fetch(agent_id, memory_type="episodic")
        if not episodic:
            return None

        if limit is not None:
            source = episodic[-limit:]
        elif len(episodic) > self._max_episodes:
            overflow = len(episodic) - self._max_episodes
            source = episodic[:overflow]
        else:
            source = episodic[-min(self._consolidation_window, len(episodic)) :]

        if not source:
            return None

        summary = await self._summarize_contents([r.content for r in source])
        if not summary.strip():
            return None

        semantic = await self._store_record(
            agent_id=agent_id,
            content=summary,
            memory_type="semantic",
            ttl_days=None,
            metadata={
                "source": "consolidation",
                "consolidated_ids": [r.id for r in source],
            },
            skip_auto_consolidate=True,
        )

        should_delete = (
            delete_consolidated
            if delete_consolidated is not None
            else len(episodic) > self._max_episodes
        )
        if should_delete:
            for rec in source:
                await self._backend.delete(agent_id, rec.id)

        return semantic

    async def delete(self, *, agent_id: str, record_id: str) -> bool:
        return await self._backend.delete(agent_id, record_id)

    async def clear(self, *, agent_id: str, memory_type: MemoryType | None = None) -> int:
        return await self._backend.clear(agent_id, memory_type=memory_type)

    async def list(
        self,
        *,
        agent_id: str,
        memory_type: MemoryType | None = None,
        include_expired: bool = False,
    ) -> list[MemoryRecord]:
        return await self._backend.fetch(
            agent_id,
            memory_type=memory_type,
            include_expired=include_expired,
        )

    async def count(self, *, agent_id: str, memory_type: MemoryType | None = None) -> int:
        return await self._backend.count(agent_id, memory_type=memory_type)
