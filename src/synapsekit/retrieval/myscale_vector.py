"""MyScale vector-search adapter."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from ..embeddings.backend import SynapsekitEmbeddings
from ._remote_vector import RemoteVectorStoreSupport
from .base import VectorStore


class MyScaleVectorStore(RemoteVectorStoreSupport, VectorStore):
    """Async adapter for MyScale's ClickHouse-compatible SQL API."""

    def __init__(
        self,
        embedding_backend: SynapsekitEmbeddings,
        host: str = "127.0.0.1",
        port: int = 443,
        username: str = "default",
        password: str = "",
        database: str = "default",
        table_name: str = "synapsekit_vectors",
        secure: bool = True,
        client: Any | None = None,
    ) -> None:
        if client is None:
            try:
                import clickhouse_connect
            except ImportError:
                raise ImportError(
                    "clickhouse-connect required: pip install synapsekit[myscale]"
                ) from None
            client = clickhouse_connect.get_client(
                host=host,
                port=port,
                username=username,
                password=password,
                database=database,
                secure=secure,
            )
        self._embeddings = embedding_backend
        self._client = client
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

    def _ensure_table(self, dim: int) -> None:
        if self._table_created and self._dim == dim:
            return
        self._client.command(
            f"CREATE TABLE IF NOT EXISTS {self._table_name} ("
            "id String, text String, metadata String, embedding Array(Float32)) "
            "ENGINE = MergeTree ORDER BY id"
        )
        self._table_created = True
        self._dim = dim

    def _add_sync(
        self, documents: list[tuple[str, dict[str, Any]]], vectors: list[list[float]]
    ) -> None:
        self._ensure_table(len(vectors[0]))
        rows = [
            [str(uuid.uuid4()), text, json.dumps(metadata, ensure_ascii=False), vector]
            for (text, metadata), vector in zip(documents, vectors, strict=True)
        ]
        self._client.insert(
            self._table_name,
            rows,
            column_names=["id", "text", "metadata", "embedding"],
        )

    def _search_sync(
        self, vector: list[float], top_k: int, metadata_filter: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        limit = top_k * 10 if metadata_filter else top_k
        exists = self._client.command(f"EXISTS TABLE {self._table_name}")
        if str(exists).strip() != "1":
            return []
        vector_literal = "[" + ",".join(str(float(value)) for value in vector) + "]"
        result = self._client.query(
            f"SELECT text, metadata, 1 - CosineDistance(embedding, {vector_literal}) AS score "
            f"FROM {self._table_name} ORDER BY score DESC LIMIT {{limit:UInt32}}",
            parameters={"limit": limit},
        )
        output = []
        for text, raw_metadata, score in result.result_rows:
            try:
                metadata = json.loads(raw_metadata) if raw_metadata else {}
            except (json.JSONDecodeError, TypeError):
                metadata = {}
            if not isinstance(metadata, dict):
                metadata = {}
            if metadata_filter and not all(
                metadata.get(key) == value for key, value in metadata_filter.items()
            ):
                continue
            output.append({"text": text, "score": float(score), "metadata": metadata})
            if len(output) == top_k:
                break
        return output

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
