"""Cohere embeddings provider (embed-v3, input-type aware)."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .base import BaseEmbeddings


class CohereEmbeddings(BaseEmbeddings):
    """Async embeddings backed by the Cohere Embed API.

    Cohere's ``embed-v3`` models are input-type aware: documents and queries
    should be embedded with ``input_type="search_document"`` and
    ``input_type="search_query"`` respectively. ``embed()`` uses the document
    type; ``embed_one()`` switches to the query type so the same instance can
    serve both ingestion and retrieval.

    Usage::

        emb = CohereEmbeddings(api_key="...")               # or CO_API_KEY
        doc_vecs = await emb.embed(["doc1", "doc2"])
        query_vec = await emb.embed_one("a question")       # search_query

    Requires ``cohere``: ``pip install synapsekit[cohere]``
    """

    dimensions: int | None = 1024

    def __init__(
        self,
        model: str = "embed-v3",
        *,
        api_key: str | None = None,
        input_type: str = "search_document",
        query_input_type: str = "search_query",
        batch_size: int = 64,
        normalize: bool = True,
    ) -> None:
        super().__init__(batch_size=batch_size, normalize=normalize)
        self.model = model
        self._api_key = api_key
        self._input_type = input_type
        self._query_input_type = query_input_type
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from cohere import AsyncClientV2
            except ImportError:
                raise ImportError("cohere required: pip install synapsekit[cohere]") from None
            key = self._api_key or os.environ.get("CO_API_KEY")
            if not key:
                raise ValueError("CO_API_KEY is not set")
            self._client = AsyncClientV2(api_key=key)
        return self._client

    async def _embed_raw(self, texts: list[str]) -> np.ndarray:
        client = self._get_client()
        resp = await client.embed(
            model=self.model,
            texts=texts,
            input_type=self._input_type,
            embedding_types=["float"],
        )
        return np.asarray(resp.embeddings.float_, dtype=np.float32)

    async def embed_one(self, text: str) -> np.ndarray:
        """Embed a single string using the query input type."""
        client = self._get_client()
        resp = await client.embed(
            model=self.model,
            texts=[text],
            input_type=self._query_input_type,
            embedding_types=["float"],
        )
        vec = np.asarray(resp.embeddings.float_[0], dtype=np.float32)
        if self._normalize:
            norm = np.linalg.norm(vec)
            if norm != 0:
                vec = vec / norm
        return vec
