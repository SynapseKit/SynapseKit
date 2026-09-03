"""Couchbase vector-search adapter."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

from ..embeddings.backend import SynapsekitEmbeddings
from ._remote_vector import RemoteVectorStoreSupport
from .base import VectorStore


class CouchbaseVectorStore(RemoteVectorStoreSupport, VectorStore):
    """Async adapter for Couchbase Search Vector Indexes."""

    def __init__(
        self,
        embedding_backend: SynapsekitEmbeddings,
        connection_string: str = "couchbase://127.0.0.1",
        username: str | None = None,
        password: str | None = None,
        bucket_name: str = "default",
        scope_name: str = "_default",
        collection_name: str = "_default",
        index_name: str = "synapsekit-vector-index",
        cluster: Any | None = None,
        collection: Any | None = None,
        search_scope: Any | None = None,
    ) -> None:
        if cluster is None or collection is None or search_scope is None:
            try:
                from couchbase.auth import PasswordAuthenticator
                from couchbase.cluster import Cluster, ClusterOptions
            except ImportError:
                raise ImportError("couchbase required: pip install synapsekit[couchbase]") from None
            if cluster is None:
                if username is None or password is None:
                    raise ValueError("username and password are required")
                cluster = Cluster(
                    connection_string,
                    ClusterOptions(PasswordAuthenticator(username, password)),
                )
                cluster.wait_until_ready(timedelta(seconds=10))
            bucket = cluster.bucket(bucket_name)
            scope = bucket.scope(scope_name)
            if collection is None:
                collection = scope.collection(collection_name)
            if search_scope is None:
                search_scope = scope

        self._embeddings = embedding_backend
        self._cluster = cluster
        self._collection = collection
        self._search_scope = search_scope
        self._index_name = index_name
        self._init_remote_state()

    def _add_sync(
        self, documents: list[tuple[str, dict[str, Any]]], vectors: list[list[float]]
    ) -> None:
        import uuid

        for (text, metadata), vector in zip(documents, vectors, strict=True):
            self._collection.upsert(
                str(uuid.uuid4()),
                {"text": text, "metadata": metadata, "embedding": vector},
            )

    def _search_sync(
        self, vector: list[float], top_k: int, metadata_filter: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        from couchbase.search import SearchRequest
        from couchbase.vector_search import VectorQuery, VectorSearch

        limit = max(top_k * 10, 100) if metadata_filter else top_k
        query = VectorQuery.create("embedding", vector, num_candidates=limit)
        request = SearchRequest.create(VectorSearch(query))
        try:
            result = self._search_scope.search(self._index_name, request)
        except Exception as exc:
            if "not found" in str(exc).lower() or "does not exist" in str(exc).lower():
                return []
            raise
        rows = result.rows() if callable(getattr(result, "rows", None)) else result
        output = []
        for row in rows or []:
            fields = row.fields() if callable(getattr(row, "fields", None)) else row
            fields = fields or {}
            metadata = fields.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            if metadata_filter and not all(
                metadata.get(key) == value for key, value in metadata_filter.items()
            ):
                continue
            score = getattr(row, "score", fields.get("score", 0.0))
            output.append(
                {
                    "text": fields.get("text", ""),
                    "score": float(score or 0.0),
                    "metadata": metadata,
                }
            )
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
