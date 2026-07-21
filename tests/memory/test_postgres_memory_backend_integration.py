"""Real PostgresMemoryBackend integration tests (testcontainers).

Boots a real Postgres and exercises the real asyncpg path: store/fetch round-trip
(including JSONB embedding + metadata), type filtering, touch, delete, clear,
count, TTL prune, persistence across reconnect, and the async-first contract.
Part of #829.
"""

from __future__ import annotations

import dataclasses
import inspect
import time
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("asyncpg")
_container_mod = pytest.importorskip("testcontainers.core.container")

from synapsekit.memory.backends.postgres import PostgresMemoryBackend  # noqa: E402
from synapsekit.memory.base import MemoryRecord  # noqa: E402

_PG_IMAGE = "postgres:16"


@pytest.fixture(scope="module")
def pg_dsn():
    import psycopg

    container = (
        _container_mod.DockerContainer(_PG_IMAGE)
        .with_env("POSTGRES_USER", "test")
        .with_env("POSTGRES_PASSWORD", "test")
        .with_env("POSTGRES_DB", "test")
        .with_exposed_ports(5432)
    )
    with container as c:
        host = c.get_container_host_ip()
        port = c.get_exposed_port(5432)
        dsn = f"postgresql://test:test@{host}:{port}/test"
        last_err: Exception | None = None
        for _ in range(60):
            try:
                psycopg.connect(dsn).close()
                break
            except Exception as exc:
                last_err = exc
                time.sleep(1)
        else:
            raise RuntimeError(f"postgres not ready: {last_err}")
        yield dsn


def _record(rid: str, agent: str, *, mtype="episodic", ttl_days=None, meta=None) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=rid,
        agent_id=agent,
        content=f"content of {rid}",
        memory_type=mtype,
        embedding=[0.1, 0.2, 0.3],
        created_at=now,
        accessed_at=now,
        access_count=0,
        ttl_days=ttl_days,
        metadata=meta or {"kind": "note"},
    )


@pytest.mark.asyncio
async def test_store_and_fetch_roundtrip(pg_dsn):
    backend = PostgresMemoryBackend(pg_dsn)
    try:
        await backend.store(_record("m1", "agent_rt", meta={"kind": "fact", "n": 5}))
        records = await backend.fetch("agent_rt")
        assert len(records) == 1
        r = records[0]
        assert r.content == "content of m1"
        assert r.embedding == [0.1, 0.2, 0.3]  # JSONB list round-trips, not chars
        assert r.metadata == {"kind": "fact", "n": 5}  # JSONB dict round-trips
    finally:
        await backend.clear("agent_rt")
        await backend.close()


@pytest.mark.asyncio
async def test_fetch_filters_by_memory_type(pg_dsn):
    backend = PostgresMemoryBackend(pg_dsn)
    try:
        await backend.store(_record("e1", "agent_ft", mtype="episodic"))
        await backend.store(_record("s1", "agent_ft", mtype="semantic"))
        episodic = await backend.fetch("agent_ft", "episodic")
        assert [r.id for r in episodic] == ["e1"]
        assert await backend.count("agent_ft") == 2
    finally:
        await backend.clear("agent_ft")
        await backend.close()


@pytest.mark.asyncio
async def test_touch_increments_access_count(pg_dsn):
    backend = PostgresMemoryBackend(pg_dsn)
    try:
        await backend.store(_record("t1", "agent_touch"))
        await backend.touch("agent_touch", "t1")
        records = await backend.fetch("agent_touch")
        assert records[0].access_count == 1
    finally:
        await backend.clear("agent_touch")
        await backend.close()


@pytest.mark.asyncio
async def test_delete_and_clear(pg_dsn):
    backend = PostgresMemoryBackend(pg_dsn)
    try:
        await backend.store(_record("d1", "agent_del"))
        await backend.store(_record("d2", "agent_del"))
        assert await backend.delete("agent_del", "d1") is True
        assert await backend.count("agent_del") == 1
        assert await backend.clear("agent_del") == 1
        assert await backend.count("agent_del") == 0
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_prune_expired(pg_dsn):
    backend = PostgresMemoryBackend(pg_dsn)
    try:
        # created_at set to the past + ttl_days=1 => already expired.
        old = datetime.now(timezone.utc) - timedelta(days=5)
        rec = _record("p1", "agent_prune", ttl_days=1)
        rec = dataclasses.replace(rec, created_at=old, accessed_at=old)
        await backend.store(rec)
        await backend.store(_record("p2", "agent_prune"))  # no ttl, survives
        pruned = await backend.prune_expired()
        assert pruned == 1
        assert await backend.count("agent_prune") == 1
    finally:
        await backend.clear("agent_prune")
        await backend.close()


@pytest.mark.asyncio
async def test_persistence_across_reconnect(pg_dsn):
    writer = PostgresMemoryBackend(pg_dsn)
    await writer.store(_record("pr1", "agent_persist"))
    await writer.close()

    reader = PostgresMemoryBackend(pg_dsn)
    try:
        records = await reader.fetch("agent_persist")
        assert [r.id for r in records] == ["pr1"]
    finally:
        await reader.clear("agent_persist")
        await reader.close()


def test_async_api_is_coroutine():
    # SynapseKit is async-first: the public IO surface must stay coroutines.
    for name in ("store", "fetch", "touch", "delete", "clear", "count", "prune_expired"):
        assert inspect.iscoroutinefunction(getattr(PostgresMemoryBackend, name))
