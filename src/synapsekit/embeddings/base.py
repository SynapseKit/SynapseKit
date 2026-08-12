"""Base contract for all SynapseKit embedding backends.

Every provider — local (sentence-transformers, ONNX) and hosted (OpenAI,
Cohere, Voyage, Jina, Gemini, Mistral, Nomic, mixedbread, Hugging Face) —
implements the same uniform async interface:

- ``embed(texts)``  -> ``(N, D)`` float32, L2-normalized rows
- ``embed_one(text)`` -> ``(D,)`` float32
- ``embed_batch(texts, batch_size)`` -> chunked ``embed``, same semantics
- ``dimensions``   -> the static output dimension of the default model

The L2-normalization guarantee matters: ``InMemoryVectorStore`` and the other
vector-store backends compute cosine similarity as a plain dot product and
therefore assume unit-length rows.
"""

from __future__ import annotations

import numpy as np


class BaseEmbeddings:
    """Uniform async embeddings contract implemented by all backends."""

    #: Static output dimension of the provider's default model. ``None`` when
    #: the dimension can only be derived at runtime from an ``embed()`` call.
    dimensions: int | None = None

    def __init__(self, *, batch_size: int = 64, normalize: bool = True) -> None:
        self._batch_size = batch_size
        self._normalize = normalize

    async def _embed_raw(self, texts: list[str]) -> np.ndarray:
        """Provider-specific embed implementation, returning ``(N, D)``.

        Subclasses must implement this. Values may be unnormalized; the base
        ``embed()`` applies L2 normalization when ``normalize=True``.
        """
        raise NotImplementedError

    async def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts, returning an ``(N, D)`` float32 array.

        Rows are L2-normalized by default. An empty input list returns an
        empty ``(0, 0)`` float32 array.
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        arr = np.asarray(await self._embed_raw(texts), dtype=np.float32)
        if self._normalize:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            arr = (arr / norms).astype(np.float32)
        return arr

    async def embed_one(self, text: str) -> np.ndarray:
        """Embed a single string, returning a ``(D,)`` float32 array."""
        arr = await self.embed([text])
        return arr[0]

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Embed ``texts`` in chunks, returning the same result as ``embed``.

        Useful for providers with per-request input limits. Chunking is
        semantically transparent: results are concatenated in input order.
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        size = self._batch_size if batch_size is None else batch_size
        chunks = [texts[i : i + size] for i in range(0, len(texts), size)]
        parts = [await self.embed(chunk) for chunk in chunks]
        return np.concatenate(parts, axis=0)
