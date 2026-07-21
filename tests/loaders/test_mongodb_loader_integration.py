"""Real MongoDBLoader integration tests against Mongo via testcontainers. Part of #829."""

from __future__ import annotations

import inspect
import time

import pytest

pytest.importorskip("pymongo")
_container_mod = pytest.importorskip("testcontainers.core.container")

from synapsekit.loaders.mongodb import MongoDBLoader  # noqa: E402

_MONGO_IMAGE = "mongo:7"
_DB = "testdb"
_COLL = "articles"


@pytest.fixture(scope="module")
def mongo_uri():
    from pymongo import MongoClient

    container = _container_mod.DockerContainer(_MONGO_IMAGE).with_exposed_ports(27017)
    with container as c:
        host = c.get_container_host_ip()
        port = c.get_exposed_port(27017)
        uri = f"mongodb://{host}:{port}/?directConnection=true"
        last_err: Exception | None = None
        for _ in range(60):
            try:
                client = MongoClient(uri, serverSelectionTimeoutMS=1000)
                client.admin.command("ping")
                break
            except Exception as exc:
                last_err = exc
                time.sleep(1)
        else:
            raise RuntimeError(f"mongo not ready: {last_err}")
        client[_DB][_COLL].insert_many(
            [
                {"title": "Apples", "content": "about fruit", "category": "food"},
                {"title": "Cars", "content": "about vehicles", "category": "auto"},
            ]
        )
        client.close()
        yield uri


def test_loads_with_text_and_metadata_fields(mongo_uri):
    docs = MongoDBLoader(
        mongo_uri, _DB, _COLL, text_fields=["title", "content"], metadata_fields=["category"]
    ).load()
    assert len(docs) == 2
    texts = {d.text for d in docs}
    assert "Apples\nabout fruit" in texts
    assert all(d.metadata["source"] == "mongodb" for d in docs)
    assert {d.metadata["category"] for d in docs} == {"food", "auto"}


def test_query_filter(mongo_uri):
    docs = MongoDBLoader(
        mongo_uri, _DB, _COLL, query_filter={"category": "auto"}, text_fields=["title"]
    ).load()
    assert [d.text for d in docs] == ["Cars"]


@pytest.mark.asyncio
async def test_aload_matches(mongo_uri):
    docs = await MongoDBLoader(mongo_uri, _DB, _COLL, text_fields=["title"]).aload()
    assert len(docs) == 2


def test_async_api_is_coroutine():
    assert inspect.iscoroutinefunction(MongoDBLoader.aload)
