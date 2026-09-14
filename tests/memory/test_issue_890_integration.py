from __future__ import annotations

import os
import re
import time
import uuid
import warnings
from collections.abc import Iterator
from datetime import datetime, timezone

import pytest

from synapsekit.graph.checkpointers.cassandra import CassandraCheckpointer
from synapsekit.graph.checkpointers.cosmos import CosmosDBCheckpointer
from synapsekit.graph.checkpointers.firestore import FirestoreCheckpointer
from synapsekit.graph.checkpointers.mongodb import MongoDBCheckpointer
from synapsekit.memory.backends.cassandra import CassandraMemoryBackend
from synapsekit.memory.backends.cosmos import CosmosDBMemoryBackend
from synapsekit.memory.backends.dynamodb import DynamoDBMemoryBackend
from synapsekit.memory.backends.firestore import FirestoreMemoryBackend
from synapsekit.memory.backends.mongodb import MongoDBMemoryBackend
from synapsekit.memory.base import MemoryRecord


def _docker_available() -> bool:
    try:
        import docker

        docker.from_env().ping()
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(not _docker_available(), reason="Docker is unavailable")


@pytest.fixture(scope="module")
def mongo_uri() -> Iterator[str]:
    from testcontainers.core.container import DockerContainer

    with DockerContainer("mongo:7").with_exposed_ports(27017) as container:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(27017)
        yield f"mongodb://{host}:{port}"


@pytest.fixture(scope="module")
def cassandra_config() -> Iterator[tuple[str, int]]:
    from cassandra.cluster import Cluster
    from testcontainers.core.container import DockerContainer

    with DockerContainer("cassandra:4.1").with_exposed_ports(9042) as container:
        host = container.get_container_host_ip()
        port = int(container.get_exposed_port(9042))
        last_error: Exception | None = None
        for _ in range(90):
            cluster: Cluster | None = None
            try:
                cluster = Cluster(contact_points=[host], port=port, connect_timeout=5)
                session = cluster.connect()
                session.shutdown()
                break
            except Exception as exc:
                last_error = exc
                time.sleep(1)
            finally:
                if cluster is not None:
                    cluster.shutdown()
        else:
            raise RuntimeError(f"Cassandra was not ready: {last_error}")
        yield host, port


@pytest.fixture(scope="module")
def dynamodb_endpoint() -> Iterator[str]:
    from testcontainers.core.container import DockerContainer

    with DockerContainer("amazon/dynamodb-local:latest").with_exposed_ports(8000) as container:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(8000)
        yield f"http://{host}:{port}"


@pytest.mark.asyncio
async def test_mongodb_memory_and_checkpointer_survive_reconnect(mongo_uri: str) -> None:
    database = f"synapsekit_{uuid.uuid4().hex}"
    record = MemoryRecord(
        id="mongo-record",
        agent_id="mongo-agent",
        content="persisted",
        memory_type="semantic",
        embedding=[0.1, 0.2],
        created_at=datetime.now(timezone.utc),
        accessed_at=datetime.now(timezone.utc),
        metadata={"source": "integration"},
    )
    first = MongoDBMemoryBackend(mongo_uri, database=database)
    await first.store(record)
    await first.aclose()

    second = MongoDBMemoryBackend(mongo_uri, database=database)
    assert await second.fetch(record.agent_id) == [record]
    assert await second.touch(record.agent_id, record.id) is None
    assert (await second.fetch(record.agent_id))[0].access_count == 1
    checkpointer = MongoDBCheckpointer(mongo_uri, database=database)
    await checkpointer.asave("graph/1", 3, {"bytes": [1, 2], "nested": True})
    assert await checkpointer.aload("graph/1") == (3, {"bytes": [1, 2], "nested": True})
    checkpointer.close()
    await second.aclose()


@pytest.mark.asyncio
async def test_dynamodb_memory_survives_reconnect(dynamodb_endpoint: str) -> None:
    table = f"memory_{uuid.uuid4().hex}"
    record = MemoryRecord(
        id="dynamo-record",
        agent_id="dynamo-agent",
        content="persisted",
        memory_type="episodic",
        embedding=[1.0, -2.0],
        created_at=datetime.now(timezone.utc),
        accessed_at=datetime.now(timezone.utc),
        ttl_days=1,
        metadata={"kind": "event"},
    )
    client_options = {
        "aws_access_key_id": "testing",
        "aws_secret_access_key": "testing",
    }
    first = DynamoDBMemoryBackend(table, endpoint_url=dynamodb_endpoint, **client_options)
    await first.store(record)
    await first.aclose()
    second = DynamoDBMemoryBackend(table, endpoint_url=dynamodb_endpoint, **client_options)
    assert await second.fetch(record.agent_id) == [record]
    assert await second.count(record.agent_id, "episodic") == 1
    await second.aclose()


@pytest.mark.asyncio
async def test_cassandra_memory_and_checkpointer_survive_reconnect(
    cassandra_config: tuple[str, int],
) -> None:
    host, port = cassandra_config
    keyspace = f"synapsekit_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc)
    record = MemoryRecord(
        id="cassandra-record",
        agent_id="cassandra-agent",
        content="persisted",
        memory_type="semantic",
        embedding=[0.5, -0.25],
        created_at=now,
        accessed_at=now,
        metadata={"kind": "fact"},
    )
    first = CassandraMemoryBackend(hosts=host, port=port, keyspace=keyspace)
    await first.store(record)
    await first.aclose()
    second = CassandraMemoryBackend(hosts=host, port=port, keyspace=keyspace)
    assert await second.fetch(record.agent_id) == [record]
    await second.aclose()
    checkpointer = CassandraCheckpointer(hosts=host, port=port, keyspace=keyspace)
    await checkpointer.asave("cassandra-graph", 4, {"state": [1, 2, 3]})
    assert await checkpointer.aload("cassandra-graph") == (4, {"state": [1, 2, 3]})
    checkpointer.close()


@pytest.fixture(scope="module")
def firestore_host() -> Iterator[str]:
    from testcontainers.core.container import DockerContainer

    previous = os.environ.get("FIRESTORE_EMULATOR_HOST")
    with DockerContainer("mtlynch/firestore-emulator:latest").with_exposed_ports(8080) as container:
        endpoint = f"{container.get_container_host_ip()}:{container.get_exposed_port(8080)}"
        os.environ["FIRESTORE_EMULATOR_HOST"] = endpoint
        try:
            yield endpoint
        finally:
            if previous is None:
                os.environ.pop("FIRESTORE_EMULATOR_HOST", None)
            else:
                os.environ["FIRESTORE_EMULATOR_HOST"] = previous


@pytest.mark.asyncio
async def test_firestore_memory_and_checkpointer_survive_reconnect(firestore_host: str) -> None:
    del firestore_host
    suffix = uuid.uuid4().hex
    now = datetime.now(timezone.utc)
    record = MemoryRecord(
        id=f"firestore-record-{suffix}",
        agent_id=f"firestore-agent-{suffix}",
        content="persisted",
        memory_type="episodic",
        embedding=[2.0, 4.0],
        created_at=now,
        accessed_at=now,
        metadata={"kind": "event"},
    )
    first = FirestoreMemoryBackend(project_id="synapsekit-test", collection=f"memory-{suffix}")
    await first.store(record)
    await first.aclose()
    second = FirestoreMemoryBackend(project_id="synapsekit-test", collection=f"memory-{suffix}")
    assert await second.fetch(record.agent_id) == [record]
    await second.aclose()
    checkpointer = FirestoreCheckpointer(
        project_id="synapsekit-test", collection=f"checkpoints-{suffix}"
    )
    await checkpointer.asave("firestore-graph", 5, {"ok": True})
    assert await checkpointer.aload("firestore-graph") == (5, {"ok": True})
    checkpointer.close()


@pytest.mark.asyncio
async def test_firestore_memory_query_methods_use_field_filter(firestore_host: str) -> None:
    """Regression test for #1032: positional `.where()` args are deprecated and
    a future google-cloud-firestore major will raise instead of warn."""
    del firestore_host
    suffix = uuid.uuid4().hex
    now = datetime.now(timezone.utc)
    agent_id = f"firestore-agent-{suffix}"
    record = MemoryRecord(
        id=f"firestore-record-{suffix}",
        agent_id=agent_id,
        content="persisted",
        memory_type="episodic",
        embedding=[1.0, 1.0],
        created_at=now,
        accessed_at=now,
        metadata={},
    )
    backend = FirestoreMemoryBackend(project_id="synapsekit-test", collection=f"memory-{suffix}")
    await backend.store(record)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert await backend.count(agent_id) == 1
        assert len(await backend.fetch(agent_id)) == 1
        assert await backend.clear(agent_id) == 1
    await backend.aclose()


@pytest.fixture(scope="module")
def cosmos_config() -> Iterator[tuple[str, str]]:
    from testcontainers.core.container import DockerContainer

    with (
        DockerContainer("mcr.microsoft.com/cosmosdb/linux/azure-cosmos-emulator:vnext-preview")
        .with_env("PROTOCOL", "https")
        .with_env("ENABLE_EXPLORER", "false")
        .with_env("AZURE_COSMOS_EMULATOR_PARTITION_COUNT", "1")
        .with_exposed_ports(8081, 1234)
    ) as container:
        stdout, stderr = container.get_logs()
        logs = (stdout + stderr).decode("utf-8", errors="replace")
        match = re.search(r"Master Key:\s*(\S+)", logs)
        if match is None:
            pytest.skip("Cosmos emulator did not expose a startup master key")
        endpoint = f"https://{container.get_container_host_ip()}:{container.get_exposed_port(1234)}"
        key = match.group(1)
        yield endpoint, key


@pytest.mark.asyncio
async def test_cosmos_memory_and_checkpointer_survive_reconnect(
    cosmos_config: tuple[str, str],
) -> None:
    endpoint, key = cosmos_config
    suffix = uuid.uuid4().hex
    now = datetime.now(timezone.utc)
    record = MemoryRecord(
        id=f"cosmos-record-{suffix}",
        agent_id=f"cosmos-agent-{suffix}",
        content="persisted",
        memory_type="semantic",
        embedding=[-1.0, 0.125],
        created_at=now,
        accessed_at=now,
        metadata={"kind": "fact"},
    )
    first = CosmosDBMemoryBackend(
        endpoint=endpoint,
        key=key,
        database=f"synapsekit-{suffix}",
        container="memory",
        verify_ssl=False,
    )
    await first.store(record)
    await first.aclose()
    second = CosmosDBMemoryBackend(
        endpoint=endpoint,
        key=key,
        database=f"synapsekit-{suffix}",
        container="memory",
        verify_ssl=False,
    )
    assert await second.fetch(record.agent_id) == [record]
    await second.aclose()
    checkpointer = CosmosDBCheckpointer(
        endpoint=endpoint,
        key=key,
        database=f"synapsekit-{suffix}",
        container="checkpoints",
        verify_ssl=False,
    )
    await checkpointer.asave("cosmos-graph", 6, {"ok": [True, False]})
    assert await checkpointer.aload("cosmos-graph") == (6, {"ok": [True, False]})
    checkpointer.close()
