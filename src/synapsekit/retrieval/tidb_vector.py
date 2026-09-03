"""TiDB Vector Search adapter."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any
from urllib.parse import unquote, urlparse

from ..embeddings.backend import SynapsekitEmbeddings
from ._remote_vector import RemoteVectorStoreSupport
from .base import VectorStore


class TiDBVectorStore(RemoteVectorStoreSupport, VectorStore):
    """Async adapter for TiDB's MySQL-compatible vector SQL functions."""

    def __init__(
        self,
        embedding_backend: SynapsekitEmbeddings,
        connection_string: str | None = None,
        table_name: str = "synapsekit_vectors",
        host: str = "127.0.0.1",
        port: int = 4000,
        user: str = "root",
        password: str = "",
        database: str = "test",
        connection: Any | None = None,
    ) -> None:
        if connection is None:
            try:
                import pymysql
            except ImportError:
                raise ImportError("pymysql required: pip install synapsekit[tidb]") from None
            if connection_string:
                parsed = urlparse(connection_string)
                connection = pymysql.connect(
                    host=parsed.hostname or host,
                    port=parsed.port or port,
                    user=unquote(parsed.username or user),
                    password=unquote(parsed.password or password),
                    database=(parsed.path.lstrip("/") or database),
                    autocommit=True,
                )
            else:
                connection = pymysql.connect(
                    host=host,
                    port=port,
                    user=user,
                    password=password,
                    database=database,
                    autocommit=True,
                )
        self._embeddings = embedding_backend
        self._connection = connection
        self._table_name = self._quote_identifier(table_name)
        self._table_created = False
        self._dim: int | None = None
        self._init_remote_state()

    @staticmethod
    def _quote_identifier(name: str) -> str:
        if (
            not name
            or not name.replace("_", "").isalnum()
            or not (name[0].isalpha() or name[0] == "_")
        ):
            raise ValueError(f"invalid SQL identifier: {name!r}")
        return f"`{name.replace('`', '``')}`"

    def _execute(self, statement: str, params: tuple[Any, ...] = ()) -> list[Any]:
        cursor = self._connection.cursor()
        try:
            cursor.execute(statement, params)
            rows = cursor.fetchall() if getattr(cursor, "description", None) else []
        finally:
            close = getattr(cursor, "close", None)
            if close:
                close()
        commit = getattr(self._connection, "commit", None)
        if commit:
            commit()
        return list(rows or [])

    def _ensure_table(self, dim: int) -> None:
        if self._table_created and self._dim == dim:
            return
        self._execute(
            f"CREATE TABLE IF NOT EXISTS {self._table_name} ("
            "id CHAR(36) PRIMARY KEY, text TEXT NOT NULL, metadata JSON, "
            f"embedding VECTOR({dim}) COMMENT 'hnsw(distance=cosine)')"
        )
        self._table_created = True
        self._dim = dim

    def _add_sync(
        self, documents: list[tuple[str, dict[str, Any]]], vectors: list[list[float]]
    ) -> None:
        self._ensure_table(len(vectors[0]))
        for (text, metadata), vector in zip(documents, vectors, strict=True):
            self._execute(
                f"INSERT INTO {self._table_name} "
                "(id, text, metadata, embedding) VALUES (%s, %s, %s, %s)",
                (
                    str(uuid.uuid4()),
                    text,
                    json.dumps(metadata, ensure_ascii=False),
                    json.dumps(vector),
                ),
            )

    def _search_sync(
        self, vector: list[float], top_k: int, metadata_filter: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        limit = top_k * 10 if metadata_filter else top_k
        try:
            rows = self._execute(
                f"SELECT text, metadata, 1 - VEC_COSINE_DISTANCE(embedding, %s) AS score "
                f"FROM {self._table_name} ORDER BY score DESC LIMIT %s",
                (json.dumps(vector), limit),
            )
        except Exception as exc:
            if "doesn't exist" in str(exc).lower() or "does not exist" in str(exc).lower():
                return []
            raise
        results = []
        for row in rows:
            if isinstance(row, dict):
                text, raw_metadata, score = (
                    row.get("text", ""),
                    row.get("metadata"),
                    row.get("score", 0.0),
                )
            else:
                text, raw_metadata, score = row[:3]
            try:
                metadata = (
                    json.loads(raw_metadata)
                    if isinstance(raw_metadata, (str, bytes))
                    else raw_metadata or {}
                )
            except (json.JSONDecodeError, TypeError):
                metadata = {}
            if not isinstance(metadata, dict):
                metadata = {}
            if metadata_filter and not all(
                metadata.get(key) == value for key, value in metadata_filter.items()
            ):
                continue
            results.append({"text": text, "score": float(score), "metadata": metadata})
            if len(results) == top_k:
                break
        return results

    async def add(
        self,
        texts: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        if not texts:
            return
        await self._flush_pending()
        documents = self._validate_documents(texts, metadata)
        vectors = self._as_float_lists(await self._embeddings.embed(texts))
        await asyncio.to_thread(self._add_sync, documents, vectors)
        self._remember_documents(documents)

    async def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        top_k = self._validate_top_k(top_k)
        if top_k == 0:
            return []
        await self._flush_pending()
        vector = self._as_float_list(await self._embeddings.embed_one(query))
        return await asyncio.to_thread(self._search_sync, vector, top_k, metadata_filter)
