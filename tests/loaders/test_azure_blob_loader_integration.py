"""Real AzureBlobLoader integration tests against Azurite via testcontainers. Part of #829."""

from __future__ import annotations

import inspect
import time

import pytest

pytest.importorskip("azure.storage.blob")
_container_mod = pytest.importorskip("testcontainers.core.container")

from synapsekit.loaders.azure_blob import AzureBlobLoader  # noqa: E402

_AZURITE_IMAGE = "mcr.microsoft.com/azure-storage/azurite:latest"
# Azurite's well-known dev account credentials.
_ACCOUNT = "devstoreaccount1"
_KEY = "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw=="
_CONTAINER = "docs"


def _conn_string(host: str, port: int) -> str:
    return (
        "DefaultEndpointsProtocol=http;"
        f"AccountName={_ACCOUNT};"
        f"AccountKey={_KEY};"
        f"BlobEndpoint=http://{host}:{port}/{_ACCOUNT};"
    )


@pytest.fixture(scope="module")
def azurite_conn():
    from azure.storage.blob import BlobServiceClient

    container = (
        _container_mod.DockerContainer(_AZURITE_IMAGE)
        .with_command("azurite-blob --blobHost 0.0.0.0")
        .with_exposed_ports(10000)
    )
    with container as c:
        host = c.get_container_host_ip()
        port = int(c.get_exposed_port(10000))
        cs = _conn_string(host, port)
        svc = None
        for _ in range(60):
            try:
                svc = BlobServiceClient.from_connection_string(cs)
                svc.create_container(_CONTAINER)
                break
            except Exception as exc:
                # Container may already exist once ready; treat that as success.
                if "ContainerAlreadyExists" in str(exc):
                    break
                time.sleep(1)
        else:
            raise RuntimeError("azurite did not become ready")

        container_client = svc.get_container_client(_CONTAINER)
        container_client.upload_blob("a.txt", b"hello world", overwrite=True)
        container_client.upload_blob("notes/b.md", b"# heading", overwrite=True)
        yield cs


def _loader(cs: str, **kw) -> AzureBlobLoader:
    return AzureBlobLoader(container_name=_CONTAINER, connection_string=cs, **kw)


def test_loads_blobs_with_text_and_metadata(azurite_conn):
    docs = _loader(azurite_conn).load()
    by_key = {d.metadata["blob_name"]: d for d in docs}
    assert set(by_key) >= {"a.txt", "notes/b.md"}
    a = by_key["a.txt"]
    assert a.text == "hello world"
    assert a.metadata["source"] == "azure_blob"
    assert a.metadata["container"] == _CONTAINER


def test_prefix_filter(azurite_conn):
    docs = _loader(azurite_conn, prefix="notes/").load()
    texts = {d.text for d in docs}
    assert "# heading" in texts
    assert "hello world" not in texts


@pytest.mark.asyncio
async def test_aload_matches(azurite_conn):
    docs = await _loader(azurite_conn).aload()
    assert len(docs) >= 2


def test_async_api_is_coroutine():
    assert inspect.iscoroutinefunction(AzureBlobLoader.aload)
