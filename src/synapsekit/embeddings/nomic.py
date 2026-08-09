"""Nomic embeddings provider (nomic-embed-text, open + Atlas)."""

from __future__ import annotations

import numpy as np

from .http import HTTPEmbeddings


class NomicEmbeddings(HTTPEmbeddings):
    """Async embeddings backed by the Nomic Embeddings API.

    ``nomic-embed-text`` supports a ``task_type`` field; ingestion should use
    ``search_document`` and queries ``search_query``. ``embed()`` sends the
    document type, ``embed_one()`` the query type, matching the Cohere
    input-type pattern.

    Usage::

        emb = NomicEmbeddings(api_key="nomic_...")          # or NOMIC_API_KEY
        vecs = await emb.embed(["hello", "world"])
        query_vec = await emb.embed_one("a question")       # search_query

    Requires ``httpx``: ``pip install synapsekit[nomic]``
    """

    dimensions: int | None = 768

    def __init__(
        self,
        model: str = "nomic-embed-text",
        *,
        api_key: str | None = None,
        task_type: str = "search_document",
        query_task_type: str = "search_query",
        batch_size: int = 64,
        normalize: bool = True,
        timeout: float = 60.0,
    ) -> None:
        self._task_type = task_type
        self._query_task_type = query_task_type
        super().__init__(
            model,
            api_key=api_key,
            base_url="https://api.nomic.ai/v1",
            env_key="NOMIC_API_KEY",
            batch_size=batch_size,
            normalize=normalize,
            timeout=timeout,
            task_type=task_type,
        )

    async def embed_one(self, text: str) -> np.ndarray:
        """Embed a single string using the query task type."""
        old = self._request_extra.get("task_type")
        self._request_extra["task_type"] = self._query_task_type
        try:
            arr = await self.embed([text])
            return arr[0]
        finally:
            self._request_extra["task_type"] = old
