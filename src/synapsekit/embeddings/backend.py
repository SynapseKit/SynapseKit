"""Local sentence-transformers embeddings backend."""

from __future__ import annotations

import numpy as np

from .base import BaseEmbeddings


class SynapsekitEmbeddings(BaseEmbeddings):
    """
    Async embeddings using sentence-transformers.
    Lazy-loads the model on first use.
    """

    dimensions: int | None = 384

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        use_gpu: bool = False,
        *,
        batch_size: int = 64,
        normalize: bool = True,
        dimensions: int | None = None,
    ) -> None:
        super().__init__(batch_size=batch_size, normalize=normalize)
        self.model = model
        self.use_gpu = use_gpu
        if dimensions is not None:
            self.dimensions = dimensions
        self._backend = None

    def _get_backend(self):
        if self._backend is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers required: pip install synapsekit[semantic]"
                ) from None
            device = "cuda" if self.use_gpu else "cpu"
            self._backend = SentenceTransformer(self.model, device=device)
        return self._backend

    async def _embed_raw(self, texts: list[str]) -> np.ndarray:
        """Run the sentence-transformer model on ``texts``."""
        import asyncio

        from ..observe.runtime import end_span, record_exception, start_span

        span = start_span(
            "embedding.encode",
            {
                "embedding.model": self.model,
                "embedding.batch_size": len(texts),
                "embedding.inputs": list(texts),
            },
        )
        try:
            backend = self._get_backend()
            loop = asyncio.get_event_loop()
            vecs = await loop.run_in_executor(None, backend.encode, texts)
            return np.array(vecs, dtype=np.float32)
        except Exception as exc:
            record_exception(span, exc)
            raise
        finally:
            end_span(span)

    async def embed_one(self, text: str) -> np.ndarray:
        """Embed a single string, returns (D,) float32 array."""
        arr = await self.embed([text])
        return arr[0]
