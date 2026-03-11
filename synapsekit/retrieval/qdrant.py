from __future__ import annotations

from typing import List, Optional

from ..embeddings.backend import SynapsekitEmbeddings
from .base import VectorStore


class QdrantVectorStore(VectorStore):
    """Qdrant-backed vector store. Embeds externally via SynapsekitEmbeddings."""

    def __init__(
        self,
        embedding_backend: SynapsekitEmbeddings,
        collection_name: str = "synapsekit",
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            raise ImportError("qdrant-client required: pip install synapsekit[qdrant]")

        self._embeddings = embedding_backend
        self._collection = collection_name
        self._client = QdrantClient(url=url, api_key=api_key)
        self._count = 0

    async def add(
        self,
        texts: List[str],
        metadata: Optional[List[dict]] = None,
    ) -> None:
        if not texts:
            return
        try:
            from qdrant_client.models import Distance, PointStruct, VectorParams
        except ImportError:
            raise ImportError("qdrant-client required: pip install synapsekit[qdrant]")

        meta = metadata or [{} for _ in texts]
        vecs = await self._embeddings.embed(texts)
        dim = vecs.shape[1]

        try:
            self._client.get_collection(self._collection)
        except Exception:
            self._client.create_collection(
                self._collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

        points = [
            PointStruct(
                id=self._count + i,
                vector=vecs[i].tolist(),
                payload={"text": texts[i], **meta[i]},
            )
            for i in range(len(texts))
        ]
        self._client.upsert(collection_name=self._collection, points=points)
        self._count += len(texts)

    async def search(self, query: str, top_k: int = 5) -> List[dict]:
        q_vec = await self._embeddings.embed_one(query)
        results = self._client.search(
            collection_name=self._collection,
            query_vector=q_vec.tolist(),
            limit=top_k,
        )
        return [
            {
                "text": r.payload.get("text", ""),
                "score": r.score,
                "metadata": {k: v for k, v in r.payload.items() if k != "text"},
            }
            for r in results
        ]
