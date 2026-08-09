"""Mistral embeddings provider (mistral-embed)."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .base import BaseEmbeddings


class MistralEmbeddings(BaseEmbeddings):
    """Async embeddings backed by the Mistral Embeddings API.

    Usage::

        emb = MistralEmbeddings(api_key="...")              # or MISTRAL_API_KEY
        vecs = await emb.embed(["hello", "world"])          # (2, 1024) float32

    Requires ``mistralai``: ``pip install synapsekit[mistral]``
    """

    dimensions: int | None = 1024

    def __init__(
        self,
        model: str = "mistral-embed",
        *,
        api_key: str | None = None,
        batch_size: int = 64,
        normalize: bool = True,
    ) -> None:
        super().__init__(batch_size=batch_size, normalize=normalize)
        self.model = model
        self._api_key = api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from mistralai import Mistral
            except ImportError:
                raise ImportError("mistralai required: pip install synapsekit[mistral]") from None
            key = self._api_key or os.environ.get("MISTRAL_API_KEY")
            if not key:
                raise ValueError("MISTRAL_API_KEY is not set")
            self._client = Mistral(api_key=key)
        return self._client

    async def _embed_raw(self, texts: list[str]) -> np.ndarray:
        client = self._get_client()
        resp = await client.embeddings_async(model=self.model, inputs=texts)
        ordered = sorted(resp.data, key=lambda item: item.index)
        return np.asarray([item.embedding for item in ordered], dtype=np.float32)
