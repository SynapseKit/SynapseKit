"""SurrealDB vector-search adapter."""

from __future__ import annotations

import asyncio
from typing import Any

from ..embeddings.backend import SynapsekitEmbeddings
from ._remote_vector import RemoteVectorStoreSupport
from .base import VectorStore


class SurrealDBVectorStore(RemoteVectorStoreSupport, VectorStore):
    """Async adapter for SurrealDB's parameterized vector SQL."""

    def __init__(
        self,
        embedding_backend: SynapsekitEmbeddings,
        url: str = "http://127.0.0.1:8000",
        namespace: str = "synapsekit",
        database: str = "synapsekit",
        table_name: str = "documents",
        username: str | None = None,
        password: str | None = None,
        token: str | None = None,
        client: Any | None = None,
    ) -> None:
        if (
            not table_name
            or not table_name.replace("_", "").isalnum()
            or not (table_name[0].isalpha() or table_name[0] == "_")
        ):
            raise ValueError(f"invalid SurrealDB table name: {table_name!r}")
        if client is None:
            try:
                from surrealdb import Surreal
            except ImportError:
                raise ImportError("surrealdb required: pip install synapsekit[surrealdb]") from None
            client = Surreal(url)
            client.connect()
            if token:
                client.authenticate(token)
            elif username is not None and password is not None:
                client.signin({"username": username, "password": password})
            client.use(namespace, database)
        self._embeddings = embedding_backend
        self._client = client
        self._namespace = namespace
        self._database = database
        self._table_name = table_name
        self._table_created = False
        self._dim: int | None = None
        self._init_remote_state()

    def _ensure_table(self, dim: int) -> None:
        if self._table_created and self._dim == dim:
            return
        self._client.query(
            f"DEFINE FIELD text ON TABLE {self._table_name} TYPE string; "
            f"DEFINE FIELD metadata ON TABLE {self._table_name} TYPE object; "
            f"DEFINE FIELD embedding ON TABLE {self._table_name} TYPE array<float>; "
            f"DEFINE INDEX {self._table_name}_vector ON TABLE {self._table_name} "
            f"COLUMNS embedding MTREE DIMENSION {int(dim)} DIST COSINE",
        )
        self._table_created = True
        self._dim = dim

    def _add_sync(
        self, documents: list[tuple[str, dict[str, Any]]], vectors: list[list[float]]
    ) -> None:
        self._ensure_table(len(vectors[0]))
        for (text, metadata), vector in zip(documents, vectors, strict=True):
            self._client.create(
                self._table_name,
                {"text": text, "metadata": metadata, "embedding": vector},
            )

    @staticmethod
    def _query_rows(response: Any) -> list[dict[str, Any]]:
        if isinstance(response, dict):
            response = response.get("result", [])
        if isinstance(response, list):
            rows: list[dict[str, Any]] = []
            for item in response:
                if isinstance(item, dict) and isinstance(item.get("result"), list):
                    rows.extend(row for row in item["result"] if isinstance(row, dict))
                elif isinstance(item, dict):
                    rows.append(item)
            return rows
        return []

    def _search_sync(
        self, vector: list[float], top_k: int, metadata_filter: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        limit = top_k * 10 if metadata_filter else top_k
        try:
            response = self._client.query(
                f"SELECT text, metadata, "
                f"vector::similarity::cosine(embedding, $query) AS score "
                f"FROM {self._table_name} ORDER BY score DESC LIMIT $limit",
                {"query": vector, "limit": limit},
            )
        except Exception as exc:
            if "not found" in str(exc).lower() or "does not exist" in str(exc).lower():
                return []
            raise
        results = []
        for row in self._query_rows(response):
            metadata = row.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}
            if metadata_filter and not all(
                metadata.get(key) == value for key, value in metadata_filter.items()
            ):
                continue
            results.append(
                {
                    "text": row.get("text", ""),
                    "score": float(row.get("score", 0.0) or 0.0),
                    "metadata": metadata,
                }
            )
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
