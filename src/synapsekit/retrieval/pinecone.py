from __future__ import annotations

from typing import List, Optional

from ..embeddings.backend import SynapsekitEmbeddings
from .base import VectorStore


class PineconeVectorStore(VectorStore):
    """Pinecone-backed vector store. Embeds externally via SynapsekitEmbeddings."""

    def __init__(
        self,
        embedding_backend: SynapsekitEmbeddings,
        index_name: str,
        api_key: str,
        environment: str = "us-east-1",
    ) -> None:
        try:
            from pinecone import Pinecone
        except ImportError:
            raise ImportError("pinecone required: pip install synapsekit[pinecone]")

        self._embeddings = embedding_backend
        self._pc = Pinecone(api_key=api_key)
        self._index = self._pc.Index(index_name)
        self._count = 0

    async def add(
        self,
        texts: List[str],
        metadata: Optional[List[dict]] = None,
    ) -> None:
        if not texts:
            return
        meta = metadata or [{} for _ in texts]
        vecs = await self._embeddings.embed(texts)
        vectors = [
            (f"vec-{self._count + i}", vecs[i].tolist(), {"text": texts[i], **meta[i]})
            for i in range(len(texts))
        ]
        self._index.upsert(vectors=vectors)
        self._count += len(texts)

    async def search(self, query: str, top_k: int = 5) -> List[dict]:
        q_vec = await self._embeddings.embed_one(query)
        results = self._index.query(
            vector=q_vec.tolist(), top_k=top_k, include_metadata=True
        )
        return [
            {
                "text": m.metadata.get("text", ""),
                "score": m.score,
                "metadata": {k: v for k, v in m.metadata.items() if k != "text"},
            }
            for m in results.matches
        ]
