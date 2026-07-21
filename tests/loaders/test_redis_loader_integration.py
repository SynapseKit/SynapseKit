"""Real RedisLoader integration tests against Redis via testcontainers. Part of #829."""

from __future__ import annotations

import inspect
import time

import pytest

pytest.importorskip("redis")
_container_mod = pytest.importorskip("testcontainers.core.container")

from synapsekit.loaders.redis_loader import RedisLoader  # noqa: E402

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
                client = redis_sync.from_url(url, decode_responses=True)
                client.ping()
                break
            except Exception as exc:
                last_err = exc
                time.sleep(1)
        else:
            raise RuntimeError(f"redis not ready: {last_err}")
        client.set("str:1", "hello world")
        client.set("str:2", "second")
        client.hset("hash:1", mapping={"a": "1", "b": "2"})
        client.set("json:1", '{"k": "v"}')
        client.close()
        yield url


def test_loads_string_values(redis_url):
    docs = RedisLoader(redis_url, pattern="str:*", value_type="string").load()
    assert {d.text for d in docs} == {"hello world", "second"}
    assert all(d.metadata["source"] == "redis" for d in docs)


def test_loads_hash_values(redis_url):
    docs = RedisLoader(redis_url, pattern="hash:*", value_type="hash").load()
    assert len(docs) == 1
    assert "a: 1" in docs[0].text and "b: 2" in docs[0].text


def test_loads_json_values(redis_url):
    docs = RedisLoader(redis_url, pattern="json:*", value_type="json").load()
    assert docs[0].text == '{"k": "v"}'


def test_limit(redis_url):
    docs = RedisLoader(redis_url, pattern="str:*", value_type="string", limit=1).load()
    assert len(docs) == 1


@pytest.mark.asyncio
async def test_aload_matches(redis_url):
    docs = await RedisLoader(redis_url, pattern="str:*", value_type="string").aload()
    assert len(docs) == 2


def test_async_api_is_coroutine():
    assert inspect.iscoroutinefunction(RedisLoader.aload)
