from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..embeddings.backend import SynapsekitEmbeddings


class InMemoryVectorStore:
    """
    Numpy-backed in-memory vector store.
    Supports cosine similarity search, save/load via .npz.
    """

    def __init__(self, embedding_backend: SynapsekitEmbeddings) -> None:
        self._embeddings = embedding_backend
        self._vectors: Optional[np.ndarray] = None   # (N, D)
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

        if self._vectors is None:
            self._vectors = vecs
        else:
            self._vectors = np.concatenate([self._vectors, vecs], axis=0)

        self._texts.extend(texts)
        self._metadata.extend(meta)

    async def search(self, query: str, top_k: int = 5) -> List[dict]:
        """Returns top_k results sorted by cosine similarity (desc)."""
        if self._vectors is None or len(self._texts) == 0:
            return []

        q_vec = await self._embeddings.embed_one(query)  # (D,)
        scores = self._vectors @ q_vec  # (N,) cosine sim (vecs are L2-normalised)

        k = min(top_k, len(self._texts))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [
            {
                "text": self._texts[i],
                "score": float(scores[i]),
                "metadata": self._metadata[i],
            }
            for i in top_indices
        ]

    def save(self, path: str) -> None:
        """Persist vectors, texts, and metadata to a .npz file."""
        if self._vectors is None:
            raise ValueError("Nothing to save — store is empty.")
        import json
        np.savez(
            path,
            vectors=self._vectors,
            texts=np.array(self._texts, dtype=object),
            metadata=np.array(
                [json.dumps(m) for m in self._metadata], dtype=object
            ),
        )

    def load(self, path: str) -> None:
        """Load vectors, texts, and metadata from a .npz file."""
        import json
        data = np.load(path, allow_pickle=True)
        self._vectors = data["vectors"].astype(np.float32)
        self._texts = list(data["texts"])
        self._metadata = [json.loads(s) for s in data["metadata"]]

    def __len__(self) -> int:
        return len(self._texts)
