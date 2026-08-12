"""Google Gemini embeddings provider (text-embedding-004)."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .base import BaseEmbeddings


class GeminiEmbeddings(BaseEmbeddings):
    """Async embeddings backed by the Google Gemini Embeddings API.

    Usage::

        emb = GeminiEmbeddings(api_key="AIza...")           # or GEMINI_API_KEY
        vecs = await emb.embed(["hello", "world"])          # (2, 768) float32

    Requires ``google-genai``: ``pip install synapsekit[gemini]``
    """

    dimensions: int | None = 768

    def __init__(
        self,
        model: str = "text-embedding-004",
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
                from google import genai
            except ImportError:
                raise ImportError("google-genai required: pip install synapsekit[gemini]") from None
            key = self._api_key or os.environ.get("GEMINI_API_KEY")
            if not key:
                raise ValueError("GEMINI_API_KEY is not set")
            self._client = genai.Client(api_key=key)
        return self._client

    async def _embed_raw(self, texts: list[str]) -> np.ndarray:
        client = self._get_client()
        # ``client.models.embed_content`` is synchronous in google-genai;
        # the awaitable surface lives under ``client.aio``.
        resp = await client.aio.models.embed_content(
            model=self.model,
            contents=texts,
        )
        return np.asarray(
            [item.values for item in resp.embeddings],
            dtype=np.float32,
        )
