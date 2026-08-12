"""OpenAI embeddings provider (text-embedding-3-small/large)."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .base import BaseEmbeddings


class OpenAIEmbeddings(BaseEmbeddings):
    """Async embeddings backed by the OpenAI Embeddings API.

    Usage::

        emb = OpenAIEmbeddings(api_key="sk-...")            # or OPENAI_API_KEY
        vecs = await emb.embed(["hello", "world"])          # (2, 1536) float32

    Requires ``openai``: ``pip install synapsekit[openai]``
    """

    dimensions: int | None = 1536

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        api_key: str | None = None,
        dimensions: int | None = None,
        batch_size: int = 64,
        normalize: bool = True,
    ) -> None:
        super().__init__(batch_size=batch_size, normalize=normalize)
        self.model = model
        self._api_key = api_key
        if dimensions is not None:
            self.dimensions = dimensions
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError(
                    "openai package required: pip install synapsekit[openai]"
                ) from None
            key = self._api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError("OPENAI_API_KEY is not set")
            self._client = AsyncOpenAI(api_key=key)
        return self._client

    async def _embed_raw(self, texts: list[str]) -> np.ndarray:
        client = self._get_client()
        kwargs: dict[str, Any] = {"model": self.model, "input": texts}
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        resp = await client.embeddings.create(**kwargs)
        ordered = sorted(resp.data, key=lambda item: item.index)
        return np.asarray([item.embedding for item in ordered], dtype=np.float32)
