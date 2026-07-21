"""Real DynamoDBLoader integration tests against DynamoDB Local via testcontainers.

Validates the new ``endpoint_url`` support (for DynamoDB Local / LocalStack) and
the real boto3 scan path into Documents. Part of #829.
"""

from __future__ import annotations

import inspect
import time

import pytest

pytest.importorskip("boto3")
_container_mod = pytest.importorskip("testcontainers.core.container")

from synapsekit.loaders.dynamodb import DynamoDBLoader  # noqa: E402

_DDB_IMAGE = "amazon/dynamodb-local:latest"
_TABLE = "articles"


@pytest.fixture(scope="module")
def ddb_endpoint():
    import boto3

    container = _container_mod.DockerContainer(_DDB_IMAGE).with_exposed_ports(8000)
    with container as c:
        host = c.get_container_host_ip()
        port = c.get_exposed_port(8000)
        endpoint = f"http://{host}:{port}"
        resource = None
        for _ in range(60):
            try:
                resource = boto3.resource(
                    "dynamodb",
                    endpoint_url=endpoint,
                    region_name="us-east-1",
                    aws_access_key_id="dummy",
                    aws_secret_access_key="dummy",
                )
                list(resource.tables.all())
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("dynamodb-local did not become ready")

        table = resource.create_table(
            TableName=_TABLE,
            KeySchema=[{"AttributeName": "id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        table.wait_until_exists()
        table.put_item(
            Item={"id": "1", "title": "Apples", "content": "about fruit", "category": "food"}
        )
        table.put_item(
            Item={"id": "2", "title": "Cars", "content": "about vehicles", "category": "auto"}
        )
        yield endpoint


def _loader(endpoint: str, **kw) -> DynamoDBLoader:
    return DynamoDBLoader(
        _TABLE,
        endpoint_url=endpoint,
        aws_access_key_id="dummy",
        aws_secret_access_key="dummy",
        **kw,
    )


def test_scan_with_text_and_metadata(ddb_endpoint):
    docs = _loader(
        ddb_endpoint, text_attributes=["title", "content"], metadata_attributes=["category"]
    ).load()
    assert len(docs) == 2
    texts = {d.text for d in docs}
    assert "Apples\nabout fruit" in texts
    assert all(d.metadata["source"] == "dynamodb" for d in docs)
    assert {d.metadata["category"] for d in docs} == {"food", "auto"}


def test_max_items(ddb_endpoint):
    docs = _loader(ddb_endpoint, text_attributes=["title"], max_items=1).load()
    assert len(docs) == 1


@pytest.mark.asyncio
async def test_aload_matches(ddb_endpoint):
    docs = await _loader(ddb_endpoint, text_attributes=["title"]).aload()
    assert len(docs) == 2


def test_async_api_is_coroutine():
    assert inspect.iscoroutinefunction(DynamoDBLoader.aload)
