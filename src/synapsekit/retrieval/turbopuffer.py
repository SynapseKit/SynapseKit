"""Turbopuffer vector-store adapter."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from ..embeddings.backend import SynapsekitEmbeddings
from ._remote_vector import RemoteVectorStoreSupport
from .base import VectorStore


class TurbopufferVectorStore(RemoteVectorStoreSupport, VectorStore):
    """Async adapter for Turbopuffer's namespace API."""

    def __init__(
        self,
        embedding_backend: SynapsekitEmbeddings,
        namespace: str = "synapsekit",
        api_key: str | None = None,
        region: str = "gcp-us-central1",
        distance_metric: str = "cosine_distance",
        client: Any | None = None,
    ) -> None:
        if client is None:
            try:
                import turbopuffer
            except ImportError:
                raise ImportError(
                    "turbopuffer required: pip install synapsekit[turbopuffer]"
                ) from None
            client = turbopuffer.Turbopuffer(api_key=api_key, region=region)
        self._embeddings = embedding_backend
        self._namespace_name = namespace
        self._distance_metric = distance_metric
        self._client = client
        self._namespace = client.namespace(namespace)
        self._init_remote_state()

    @staticmethod
    def _row_value(row: Any, key: str, default: Any = None) -> Any:
        if isinstance(row, dict):
            return row.get(key, default)
        return getattr(row, key, default)

    @classmethod
    def _response_rows(cls, response: Any) -> list[Any]:
        rows = cls._row_value(response, "rows")
        if rows is None and isinstance(response, dict):
            rows = response.get("data", {}).get("rows", [])
        return list(rows or [])

    def _add_sync(
        self, documents: list[tuple[str, dict[str, Any]]], vectors: list[list[float]]
    ) -> None:
        rows = [
            {
                "id": str(uuid.uuid4()),
                "vector": vector,
                "attributes": {
                    "text": text,
                    "metadata": json.dumps(metadata, ensure_ascii=False),
                },
            }
            for (text, metadata), vector in zip(documents, vectors, strict=True)
        ]
        self._namespace.write(
            distance_metric=self._distance_metric,
            upsert_rows=rows,
        )

    def _search_sync(
        self, vector: list[float], top_k: int, metadata_filter: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        limit = max(top_k * 10, 100) if metadata_filter else top_k
        response = self._namespace.query(
            vector=vector,
            top_k=limit,
            include_attributes=True,
        )
        results: list[dict[str, Any]] = []
        for row in self._response_rows(response):
            attributes = self._row_value(row, "attributes", {}) or {}
            raw_metadata = (
                attributes.get("metadata", "{}") if isinstance(attributes, dict) else "{}"
            )
            try:
                metadata = (
                    json.loads(raw_metadata) if isinstance(raw_metadata, str) else raw_metadata
                )
            except (json.JSONDecodeError, TypeError):
                metadata = {}
            if not isinstance(metadata, dict):
                metadata = {}
            if metadata_filter and not all(
                metadata.get(key) == value for key, value in metadata_filter.items()
            ):
                continue
            score = self._row_value(
                row,
                "dist",
                self._row_value(row, "distance", self._row_value(row, "score", 0.0)),
            )
            results.append(
                {
                    "text": attributes.get("text", "") if isinstance(attributes, dict) else "",
                    "score": float(score),
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
