from __future__ import annotations

from typing import List

import numpy as np


class SynapsekitEmbeddings:
    """
    Async embeddings using sentence-transformers.
    Lazy-loads the model on first use.
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2", use_gpu: bool = False) -> None:
        self.model = model
        self.use_gpu = use_gpu
        self._backend = None

    def _get_backend(self):
        if self._backend is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers required: pip install synapsekit[semantic]"
                )
            device = "cuda" if self.use_gpu else "cpu"
            self._backend = SentenceTransformer(self.model, device=device)
        return self._backend

    async def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts, returns (N, D) float32 array."""
        import asyncio
        backend = self._get_backend()
        loop = asyncio.get_event_loop()
        vecs = await loop.run_in_executor(None, backend.encode, texts)
        arr = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return arr / norms

    async def embed_one(self, text: str) -> np.ndarray:
        """Embed a single string, returns (D,) float32 array."""
        arr = await self.embed([text])
        return arr[0]


