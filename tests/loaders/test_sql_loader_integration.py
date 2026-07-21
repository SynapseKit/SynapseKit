"""Real SQLLoader integration tests against Postgres via testcontainers.

Boots a real Postgres, seeds a table, and exercises the real SQLAlchemy path:
rows -> Documents with text_columns/metadata_columns selection, all-columns
default, empty result, sync load, and the async-first contract. Part of #829.
"""

from __future__ import annotations

import inspect
import time

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("psycopg")
_container_mod = pytest.importorskip("testcontainers.core.container")

from synapsekit.loaders.sql import SQLLoader  # noqa: E402

_PG_IMAGE = "postgres:16"


@pytest.fixture(scope="module")
def sa_url():
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
                conn = psycopg.connect(dsn, autocommit=True)
                break
            except Exception as exc:
                last_err = exc
                time.sleep(1)
        else:
            raise RuntimeError(f"postgres not ready: {last_err}")
        conn.execute("CREATE TABLE articles (id INT, title TEXT, content TEXT, category TEXT)")
        conn.execute(
            "INSERT INTO articles VALUES "
            "(1, 'Apples', 'about fruit', 'food'),"
            "(2, 'Cars', 'about vehicles', 'auto')"
        )
        conn.close()
        yield f"postgresql+psycopg://test:test@{host}:{port}/test"


def test_selects_text_and_metadata_columns(sa_url):
    docs = SQLLoader(
        connection_string=sa_url,
        query="SELECT * FROM articles ORDER BY id",
        text_columns=["title", "content"],
        metadata_columns=["id", "category"],
    ).load()
    assert len(docs) == 2
    assert docs[0].text == "Apples about fruit"
    assert docs[0].metadata["category"] == "food"
    assert docs[0].metadata["id"] == 1
    assert docs[0].metadata["source"] == "sql"
    assert docs[0].metadata["row_index"] == 0


def test_all_columns_default(sa_url):
    docs = SQLLoader(
        connection_string=sa_url,
        query="SELECT title, category FROM articles WHERE id = 1",
    ).load()
    assert docs[0].text == "Apples food"
    assert docs[0].metadata["title"] == "Apples"


def test_empty_result(sa_url):
    docs = SQLLoader(
        connection_string=sa_url,
        query="SELECT * FROM articles WHERE id = 999",
    ).load()
    assert docs == []


@pytest.mark.asyncio
async def test_aload_matches(sa_url):
    docs = await SQLLoader(connection_string=sa_url, query="SELECT * FROM articles").aload()
    assert len(docs) == 2


def test_async_api_is_coroutine():
    # SynapseKit is async-first: aload must stay a coroutine.
    assert inspect.iscoroutinefunction(SQLLoader.aload)
