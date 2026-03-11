from __future__ import annotations

from typing import List, Optional

from ..embeddings.backend import SynapsekitEmbeddings
from .base import VectorStore


class ChromaVectorStore(VectorStore):
    """Chroma-backed vector store. Embeds externally via SynapsekitEmbeddings."""

    def __init__(
        self,
        embedding_backend: SynapsekitEmbeddings,
        collection_name: str = "synapsekit",
        persist_directory: Optional[str] = None,
    ) -> None:
        try:
            import chromadb
        except ImportError:
            raise ImportError("chromadb required: pip install synapsekit[chroma]")

        self._embeddings = embedding_backend
        self._collection_name = collection_name

        if persist_directory:
            client = chromadb.PersistentClient(path=persist_directory)
        else:
            client = chromadb.EphemeralClient()

        self._collection = client.get_or_create_collection(collection_name)
        self._offset = 0

    async def add(
        self,
        texts: List[str],
        metadata: Optional[List[dict]] = None,
    ) -> None:
        if not texts:
            return
        meta = metadata or [{} for _ in texts]
        vecs = await self._embeddings.embed(texts)
        ids = [str(self._offset + i) for i in range(len(texts))]
        self._collection.add(
            embeddings=vecs.tolist(),
            documents=texts,
            metadatas=meta,
            ids=ids,
        )
        self._offset += len(texts)

    async def search(self, query: str, top_k: int = 5) -> List[dict]:
        count = self._collection.count()
        if count == 0:
            return []
        q_vec = await self._embeddings.embed_one(query)
        results = self._collection.query(
            query_embeddings=[q_vec.tolist()],
            n_results=min(top_k, count),
        )
        out = []
        for i, doc in enumerate(results["documents"][0]):
            out.append({
                "text": doc,
                "score": 1.0 - results["distances"][0][i],
                "metadata": results["metadatas"][0][i],
            })
        return out
