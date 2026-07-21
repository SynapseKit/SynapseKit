from __future__ import annotations

import asyncio
import uuid

from ..embeddings.backend import SynapsekitEmbeddings
from .base import VectorStore


class QdrantVectorStore(VectorStore):
    """Qdrant-backed vector store. Embeds externally via SynapsekitEmbeddings.

    The collection is created on the first ``add()`` using the embedding
    dimension of the first vector; ``search()`` before any documents exist
    returns ``[]``. Point ids are UUIDs, so reconnecting and adding again
    appends rather than overwriting.
    """

    def __init__(
        self,
        embedding_backend: SynapsekitEmbeddings,
        collection_name: str = "synapsekit",
        url: str = "http://localhost:6333",
        api_key: str | None = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            raise ImportError("qdrant-client required: pip install synapsekit[qdrant]") from None

        self._embeddings = embedding_backend
        self._collection = collection_name
        self._client = QdrantClient(url=url, api_key=api_key)

    async def add(
        self,
        texts: list[str],
        metadata: list[dict] | None = None,
    ) -> None:
        if not texts:
            return
        from qdrant_client.models import Distance, PointStruct, VectorParams

        meta = metadata or [{} for _ in texts]
        if len(meta) != len(texts):
            raise ValueError("metadata must match texts length")

        vecs = await self._embeddings.embed(texts)
        dim = int(vecs.shape[1])

        def _write() -> None:
            # Blocking qdrant client — run off the event loop to keep async alive.
            if not self._client.collection_exists(self._collection):
                self._client.create_collection(
                    self._collection,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vecs[i].tolist(),
                    payload={"text": texts[i], **meta[i]},
                )
                for i in range(len(texts))
            ]
            self._client.upsert(collection_name=self._collection, points=points)

        await asyncio.to_thread(_write)

    async def search(
        self, query: str, top_k: int = 5, metadata_filter: dict | None = None
    ) -> list[dict]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        q_vec = await self._embeddings.embed_one(query)
        query_filter = None
        if metadata_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(key=key, match=MatchValue(value=value))
                    for key, value in metadata_filter.items()
                ]
            )

        def _query() -> list:
            if not self._client.collection_exists(self._collection):
                return []
            response = self._client.query_points(
                collection_name=self._collection,
                query=q_vec.tolist(),
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
            return response.points

        points = await asyncio.to_thread(_query)
        return [
            {
                "text": (p.payload or {}).get("text", ""),
                "score": p.score,
                "metadata": {k: v for k, v in (p.payload or {}).items() if k != "text"},
            }
            for p in points
        ]
