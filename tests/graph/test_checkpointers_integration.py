"""Real graph checkpointer integration tests (testcontainers).

Boots real Postgres and Redis and exercises the real client paths for
PostgresCheckpointer (psycopg + JSONB) and RedisCheckpointer: save/load
round-trip, upsert overwrite, missing -> None, delete, and persistence across a
reconnect. Part of #829.
"""

from __future__ import annotations

import time

import pytest

_container_mod = pytest.importorskip("testcontainers.core.container")

_PG_IMAGE = "postgres:16"
_REDIS_IMAGE = "redis:7"


# --------------------------------------------------------------------------- #
# Postgres
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def pg_dsn():
    pytest.importorskip("psycopg")
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


def _pg_checkpointer(dsn):
    import psycopg

    from synapsekit.graph.checkpointers.postgres import PostgresCheckpointer

    return PostgresCheckpointer(psycopg.connect(dsn))


def test_postgres_save_load_roundtrip(pg_dsn):
    cp = _pg_checkpointer(pg_dsn)
    try:
        cp.save("g_rt", 3, {"messages": ["a", "b"], "n": 7})
        result = cp.load("g_rt")
        assert result is not None
        step, state = result
        assert step == 3
        assert state == {"messages": ["a", "b"], "n": 7}  # JSONB round-trips to a dict
    finally:
        cp.delete("g_rt")
        cp.close()


def test_postgres_upsert_overwrites(pg_dsn):
    cp = _pg_checkpointer(pg_dsn)
    try:
        cp.save("g_up", 1, {"v": 1})
        cp.save("g_up", 5, {"v": 2})
        assert cp.load("g_up") == (5, {"v": 2})
    finally:
        cp.delete("g_up")
        cp.close()


def test_postgres_load_missing_returns_none(pg_dsn):
    cp = _pg_checkpointer(pg_dsn)
    try:
        assert cp.load("nope") is None
    finally:
        cp.close()


def test_postgres_delete(pg_dsn):
    cp = _pg_checkpointer(pg_dsn)
    try:
        cp.save("g_del", 2, {"x": 1})
        cp.delete("g_del")
        assert cp.load("g_del") is None
    finally:
        cp.close()


def test_postgres_persistence_across_reconnect(pg_dsn):
    writer = _pg_checkpointer(pg_dsn)
    writer.save("g_persist", 4, {"kept": True})
    writer.close()

    reader = _pg_checkpointer(pg_dsn)
    try:
        assert reader.load("g_persist") == (4, {"kept": True})
    finally:
        reader.delete("g_persist")
        reader.close()


# --------------------------------------------------------------------------- #
# Redis
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def redis_url():
    pytest.importorskip("redis")
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


def _redis_checkpointer(url, **kw):
    import redis as redis_sync

    from synapsekit.graph.checkpointers.redis import RedisCheckpointer

    return RedisCheckpointer(redis_sync.from_url(url), **kw)


def test_redis_save_load_roundtrip(redis_url):
    cp = _redis_checkpointer(redis_url)
    try:
        cp.save("g_rt", 3, {"messages": ["a", "b"], "n": 7})
        assert cp.load("g_rt") == (3, {"messages": ["a", "b"], "n": 7})
    finally:
        cp.delete("g_rt")
        cp.close()


def test_redis_load_missing_returns_none(redis_url):
    cp = _redis_checkpointer(redis_url)
    try:
        assert cp.load("nope") is None
    finally:
        cp.close()


def test_redis_delete(redis_url):
    cp = _redis_checkpointer(redis_url)
    try:
        cp.save("g_del", 2, {"x": 1})
        cp.delete("g_del")
        assert cp.load("g_del") is None
    finally:
        cp.close()


def test_redis_persistence_across_reconnect(redis_url):
    writer = _redis_checkpointer(redis_url)
    writer.save("g_persist", 4, {"kept": True})
    writer.close()

    reader = _redis_checkpointer(redis_url)
    try:
        assert reader.load("g_persist") == (4, {"kept": True})
    finally:
        reader.delete("g_persist")
        reader.close()
