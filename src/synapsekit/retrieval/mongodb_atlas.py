"""MongoDBAtlasVectorStore — Atlas Vector Search backend."""

from __future__ import annotations

import asyncio
from typing import Any

from ..embeddings.backend import SynapsekitEmbeddings
from .base import VectorStore


class MongoDBAtlasVectorStore(VectorStore):
    """MongoDB Atlas Vector Search-backed vector store."""

    def __init__(
        self,
        embedding_backend: SynapsekitEmbeddings,
        uri: str = "mongodb://localhost:27017",
        database_name: str = "synapsekit",
        collection_name: str = "documents",
        index_name: str = "vector_index",
        vector_field: str = "embedding",
        text_field: str = "text",
        metadata_field: str = "metadata",
        num_candidates_multiplier: int = 10,
        client: Any | None = None,
    ) -> None:
        if client is None:
            try:
                from pymongo import MongoClient
            except ImportError:
                raise ImportError(
                    "pymongo required: pip install synapsekit[mongodb-vector]"
                ) from None
            self._client = MongoClient(uri)
        else:
            self._client = client

        self._embeddings = embedding_backend
        self._index_name = index_name
        self._vector_field = vector_field
        self._text_field = text_field
        self._metadata_field = metadata_field
        self._num_candidates_multiplier = max(1, num_candidates_multiplier)

        self._collection = self._client[database_name][collection_name]

    def _normalize_metadata_filter(
        self, metadata_filter: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        if not metadata_filter:
            return None

        # Raw MQL operators ($and/$or/...) are passed through unchanged.
        if any(str(key).startswith("$") for key in metadata_filter):
            return metadata_filter

        # For simple dict filters, target metadata.* by default.
        return {
            key if "." in key else f"{self._metadata_field}.{key}": value
            for key, value in metadata_filter.items()
        }

    async def add(
        self,
        texts: list[str],
        metadata: list[dict] | None = None,
    ) -> None:
        if not texts:
            return

        meta = metadata or [{} for _ in texts]
        if len(meta) != len(texts):
            raise ValueError("metadata must match texts length")

        vectors = await self._embeddings.embed(texts)
        docs = [
            {
                self._text_field: text,
                self._vector_field: vector.tolist(),
                self._metadata_field: item_meta,
            }
            for text, vector, item_meta in zip(texts, vectors, meta, strict=True)
        ]
        await asyncio.to_thread(self._collection.insert_many, docs)

    async def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        if top_k <= 0:
            return []

        q_vec = await self._embeddings.embed_one(query)
        vector_search_stage: dict[str, Any] = {
            "index": self._index_name,
            "path": self._vector_field,
            "queryVector": q_vec.tolist(),
            "numCandidates": max(top_k * self._num_candidates_multiplier, top_k),
            "limit": top_k,
        }

        mql_filter = self._normalize_metadata_filter(metadata_filter)
        if mql_filter:
            vector_search_stage["filter"] = mql_filter

        pipeline = [
            {"$vectorSearch": vector_search_stage},
            {
                "$project": {
                    self._text_field: 1,
                    self._metadata_field: 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        def _run_aggregate() -> list[dict[str, Any]]:
            return list(self._collection.aggregate(pipeline))

        docs = await asyncio.to_thread(_run_aggregate)
        out: list[dict] = []
        for doc in docs:
            metadata_obj = doc.get(self._metadata_field, {})
            if not isinstance(metadata_obj, dict):
                metadata_obj = {}
            out.append(
                {
                    "text": doc.get(self._text_field, ""),
                    "score": float(doc.get("score", 0.0)),
                    "metadata": metadata_obj,
                }
            )
        return out
