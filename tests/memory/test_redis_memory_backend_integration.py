"""Real RedisMemoryBackend integration tests (testcontainers).

Boots a plain Redis and exercises the real redis.asyncio path: store/fetch
round-trip, type filtering, touch, delete, clear, count, TTL prune, persistence
across reconnect, and the async-first contract. Part of #829.
"""

from __future__ import annotations

import dataclasses
import inspect
import time
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("redis")
_container_mod = pytest.importorskip("testcontainers.core.container")

from synapsekit.memory.backends.redis import RedisMemoryBackend  # noqa: E402
from synapsekit.memory.base import MemoryRecord  # noqa: E402

_REDIS_IMAGE = "redis:7"


@pytest.fixture(scope="module")
def redis_url():
    import redis as redis_sync

    container = _container_mod.DockerContainer(_REDIS_IMAGE).with_exposed_ports(6379)
    with container as c:
        host = c.get_container_host_ip()
        port = c.get_exposed_port(6379)
        url = f"redis://{host}:{port}"
        last_err: Exception | None = None
        for _ in range(60):
            try:
                client = redis_sync.from_url(url)
                client.ping()
                client.close()
                break
            except Exception as exc:
                last_err = exc
                time.sleep(1)
        else:
            raise RuntimeError(f"redis not ready: {last_err}")
        yield url


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
async def test_store_and_fetch_roundtrip(redis_url):
    backend = RedisMemoryBackend(redis_url)
    try:
        await backend.store(_record("m1", "agent_rt", meta={"kind": "fact", "n": 5}))
        records = await backend.fetch("agent_rt")
        assert len(records) == 1
        r = records[0]
        assert r.content == "content of m1"
        assert r.embedding == [0.1, 0.2, 0.3]
        assert r.metadata == {"kind": "fact", "n": 5}
    finally:
        await backend.clear("agent_rt")
        await backend.close()


@pytest.mark.asyncio
async def test_fetch_filters_by_memory_type(redis_url):
    backend = RedisMemoryBackend(redis_url)
    try:
        await backend.store(_record("e1", "agent_ft", mtype="episodic"))
        await backend.store(_record("s1", "agent_ft", mtype="semantic"))
        assert [r.id for r in await backend.fetch("agent_ft", "episodic")] == ["e1"]
        assert await backend.count("agent_ft") == 2
    finally:
        await backend.clear("agent_ft")
        await backend.close()


@pytest.mark.asyncio
async def test_touch_increments_access_count(redis_url):
    backend = RedisMemoryBackend(redis_url)
    try:
        await backend.store(_record("t1", "agent_touch"))
        await backend.touch("agent_touch", "t1")
        assert (await backend.fetch("agent_touch"))[0].access_count == 1
    finally:
        await backend.clear("agent_touch")
        await backend.close()


@pytest.mark.asyncio
async def test_delete_and_clear(redis_url):
    backend = RedisMemoryBackend(redis_url)
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
async def test_prune_expired(redis_url):
    backend = RedisMemoryBackend(redis_url)
    try:
        old = datetime.now(timezone.utc) - timedelta(days=5)
        expired = dataclasses.replace(
            _record("p1", "agent_prune", ttl_days=1), created_at=old, accessed_at=old
        )
        await backend.store(expired)
        await backend.store(_record("p2", "agent_prune"))
        assert await backend.prune_expired() == 1
        assert await backend.count("agent_prune") == 1
    finally:
        await backend.clear("agent_prune")
        await backend.close()


@pytest.mark.asyncio
async def test_persistence_across_reconnect(redis_url):
    writer = RedisMemoryBackend(redis_url)
    await writer.store(_record("pr1", "agent_persist"))
    await writer.close()

    reader = RedisMemoryBackend(redis_url)
    try:
        assert [r.id for r in await reader.fetch("agent_persist")] == ["pr1"]
    finally:
        await reader.clear("agent_persist")
        await reader.close()


def test_async_api_is_coroutine():
    # SynapseKit is async-first: the public IO surface must stay coroutines.
    for name in ("store", "fetch", "touch", "delete", "clear", "count", "prune_expired"):
        assert inspect.iscoroutinefunction(getattr(RedisMemoryBackend, name))
