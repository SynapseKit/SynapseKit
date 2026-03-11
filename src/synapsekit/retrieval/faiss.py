from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..embeddings.backend import SynapsekitEmbeddings
from .base import VectorStore


class FAISSVectorStore(VectorStore):
    """FAISS-backed vector store. Embeds externally via SynapsekitEmbeddings."""

    def __init__(self, embedding_backend: SynapsekitEmbeddings) -> None:
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu required: pip install synapsekit[faiss]")

        self._embeddings = embedding_backend
        self._faiss = faiss
        self._index = None
        self._texts: List[str] = []
        self._metadata: List[dict] = []

    async def add(
        self,
        texts: List[str],
        metadata: Optional[List[dict]] = None,
    ) -> None:
        if not texts:
            return
        meta = metadata or [{} for _ in texts]
        vecs = await self._embeddings.embed(texts)

        if self._index is None:
            self._index = self._faiss.IndexFlatIP(vecs.shape[1])
        self._index.add(vecs.astype(np.float32))
        self._texts.extend(texts)
        self._metadata.extend(meta)

    async def search(self, query: str, top_k: int = 5) -> List[dict]:
        if self._index is None or not self._texts:
            return []
        q_vec = await self._embeddings.embed_one(query)
        k = min(top_k, len(self._texts))
        scores, indices = self._index.search(
            q_vec.reshape(1, -1).astype(np.float32), k
        )
        return [
            {
                "text": self._texts[idx],
                "score": float(scores[0][i]),
                "metadata": self._metadata[idx],
            }
            for i, idx in enumerate(indices[0])
            if idx >= 0
        ]

    def save(self, path: str) -> None:
        import json

        if self._index is None:
            raise ValueError("Nothing to save — store is empty.")
        self._faiss.write_index(self._index, path + ".faiss")
        np.save(path + "_texts.npy", np.array(self._texts, dtype=object))
        with open(path + "_meta.json", "w") as f:
            json.dump(self._metadata, f)

    def load(self, path: str) -> None:
        import json

        self._index = self._faiss.read_index(path + ".faiss")
        self._texts = list(np.load(path + "_texts.npy", allow_pickle=True))
        with open(path + "_meta.json") as f:
            self._metadata = json.load(f)
