"""Real S3Loader integration tests against MinIO (S3-compatible) via testcontainers.

Boots a real MinIO server and exercises the real boto3 path: list + download +
text extraction into Documents, prefix filtering, extension filtering, max_files,
and the async-first contract. Also validates the new ``endpoint_url`` support for
S3-compatible stores. Part of #829.
"""

from __future__ import annotations

import inspect
import time

import pytest

pytest.importorskip("boto3")
_container_mod = pytest.importorskip("testcontainers.core.container")

from synapsekit.loaders.s3 import S3Loader  # noqa: E402

_MINIO_IMAGE = "minio/minio:latest"
_ACCESS_KEY = "minioadmin"
_SECRET_KEY = "minioadmin"
_BUCKET = "test-bucket"


def _s3_client(endpoint: str):
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=_ACCESS_KEY,
        aws_secret_access_key=_SECRET_KEY,
        region_name="us-east-1",
        config=Config(s3={"addressing_style": "path"}),
    )


@pytest.fixture(scope="module")
def s3_endpoint():
    container = (
        _container_mod.DockerContainer(_MINIO_IMAGE)
        .with_env("MINIO_ROOT_USER", _ACCESS_KEY)
        .with_env("MINIO_ROOT_PASSWORD", _SECRET_KEY)
        .with_command("server /data")
        .with_exposed_ports(9000)
    )
    with container as c:
        host = c.get_container_host_ip()
        port = c.get_exposed_port(9000)
        endpoint = f"http://{host}:{port}"
        last_err: Exception | None = None
        for _ in range(60):
            try:
                client = _s3_client(endpoint)
                client.list_buckets()
                break
            except Exception as exc:
                last_err = exc
                time.sleep(1)
        else:
            raise RuntimeError(f"minio not ready: {last_err}")

        client.create_bucket(Bucket=_BUCKET)
        client.put_object(Bucket=_BUCKET, Key="a.txt", Body=b"hello world")
        client.put_object(Bucket=_BUCKET, Key="notes/b.md", Body=b"# heading\ntext")
        client.put_object(Bucket=_BUCKET, Key="notes/c.md", Body=b"more notes")
        yield endpoint


def _loader(endpoint: str, **kw) -> S3Loader:
    return S3Loader(
        bucket_name=_BUCKET,
        endpoint_url=endpoint,
        aws_access_key_id=_ACCESS_KEY,
        aws_secret_access_key=_SECRET_KEY,
        **kw,
    )


@pytest.mark.asyncio
async def test_loads_all_objects_with_text_and_metadata(s3_endpoint):
    docs = await _loader(s3_endpoint).aload()
    by_key = {d.metadata["key"]: d for d in docs}
    assert set(by_key) == {"a.txt", "notes/b.md", "notes/c.md"}
    assert by_key["a.txt"].text == "hello world"
    assert by_key["notes/b.md"].text == "# heading\ntext"
    assert by_key["a.txt"].metadata["bucket"] == _BUCKET
    assert by_key["a.txt"].metadata["source"] == "s3"
    assert by_key["a.txt"].metadata["size"] == len(b"hello world")


@pytest.mark.asyncio
async def test_prefix_filter(s3_endpoint):
    docs = await _loader(s3_endpoint, prefix="notes/").aload()
    assert {d.metadata["key"] for d in docs} == {"notes/b.md", "notes/c.md"}


@pytest.mark.asyncio
async def test_extension_filter(s3_endpoint):
    docs = await _loader(s3_endpoint, file_extensions=["txt"]).aload()
    assert {d.metadata["key"] for d in docs} == {"a.txt"}


@pytest.mark.asyncio
async def test_max_files(s3_endpoint):
    docs = await _loader(s3_endpoint, max_files=1).aload()
    assert len(docs) == 1


def test_sync_load_matches(s3_endpoint):
    docs = _loader(s3_endpoint).load()
    assert len(docs) == 3


def test_async_api_is_coroutine():
    # SynapseKit is async-first: aload must stay a coroutine.
    assert inspect.iscoroutinefunction(S3Loader.aload)
