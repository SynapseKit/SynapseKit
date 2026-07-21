"""Real ElasticsearchLoader integration tests via testcontainers. Part of #829."""

from __future__ import annotations

import inspect
import time

import pytest

pytest.importorskip("elasticsearch")
_container_mod = pytest.importorskip("testcontainers.core.container")

from synapsekit.loaders.elasticsearch import ElasticsearchLoader  # noqa: E402

_ES_IMAGE = "docker.elastic.co/elasticsearch/elasticsearch:8.15.3"
_INDEX = "articles"


@pytest.fixture(scope="module")
def es_url():
    from elasticsearch import Elasticsearch

    container = (
        _container_mod.DockerContainer(_ES_IMAGE)
        .with_env("discovery.type", "single-node")
        .with_env("xpack.security.enabled", "false")
        .with_env("ES_JAVA_OPTS", "-Xms512m -Xmx512m")
        .with_exposed_ports(9200)
    )
    with container as c:
        host = c.get_container_host_ip()
        port = c.get_exposed_port(9200)
        url = f"http://{host}:{port}"
        client = None
        for _ in range(120):
            try:
                client = Elasticsearch(hosts=[url])
                if client.ping():
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            raise RuntimeError("elasticsearch did not become ready")
        client.index(index=_INDEX, id="1", document={"content": "about fruit", "cat": "food"})
        client.index(index=_INDEX, id="2", document={"content": "about vehicles", "cat": "auto"})
        client.indices.refresh(index=_INDEX)
        yield url


def test_loads_documents(es_url):
    docs = ElasticsearchLoader(es_url, _INDEX, text_fields=["content"]).load()
    assert {d.text for d in docs} == {"about fruit", "about vehicles"}
    assert all(d.metadata["source"] == "elasticsearch" for d in docs)
    assert {d.metadata["id"] for d in docs} == {"1", "2"}


def test_query_filter(es_url):
    docs = ElasticsearchLoader(
        es_url, _INDEX, query={"match": {"cat": "auto"}}, text_fields=["content"]
    ).load()
    assert [d.text for d in docs] == ["about vehicles"]


@pytest.mark.asyncio
async def test_aload_matches(es_url):
    docs = await ElasticsearchLoader(es_url, _INDEX, text_fields=["content"]).aload()
    assert len(docs) == 2


def test_async_api_is_coroutine():
    assert inspect.iscoroutinefunction(ElasticsearchLoader.aload)
