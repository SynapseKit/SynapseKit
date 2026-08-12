"""Voyage AI reranker (rerank-2)."""

from __future__ import annotations

import os
from typing import Any

from .reranker import Reranker


class VoyageReranker(Reranker):
    """Rerank retrieval results using the Voyage AI Rerank API.

    Usage::

        reranker = VoyageReranker(retriever=retriever, model="rerank-2")
        results = await reranker.retrieve("What is RAG?", top_k=5)

    Requires ``httpx``: ``pip install synapsekit[voyage]``
    """

    def __init__(
        self,
        retriever,
        model: str = "rerank-2",
        api_key: str | None = None,
        fetch_k: int = 20,
    ) -> None:
        super().__init__(retriever, model, api_key=api_key, fetch_k=fetch_k)
        self._client: Any = None

    def _get_key(self) -> str:
        key = self._api_key or os.environ.get("VOYAGE_API_KEY")
        if not key:
            raise ValueError("VOYAGE_API_KEY is not set")
        return key

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import httpx
            except ImportError:
                raise ImportError(
                    "httpx required for VoyageReranker: pip install synapsekit[voyage]"
                ) from None
            self._client = httpx.Client(timeout=60.0)
        return self._client

    def _call(self, query: str, documents: list[str], top_n: int) -> list[dict]:
        import json

        client = self._get_client()
        resp = client.post(
            "https://api.voyageai.com/v1/rerank",
            headers={"Authorization": f"Bearer {self._get_key()}"},
            json={
                "model": self._model,
                "query": query,
                "documents": documents,
            },
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Voyage rerank request failed: HTTP {resp.status_code} {resp.text[:200]}"
            )
        data = json.loads(resp.content)
        scored = sorted(
            (
                {"text": documents[item["index"]], "relevance_score": item["relevance_score"]}
                for item in data["data"]
            ),
            key=lambda r: r["relevance_score"],
            reverse=True,
        )
        return scored[:top_n]

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[str]:
        """Retrieve candidates then rerank with the Voyage Rerank API."""
        import asyncio

        results = await self._retriever.retrieve(
            query, top_k=self._fetch_k, metadata_filter=metadata_filter
        )
        if not results:
            return []
        scored = await asyncio.to_thread(self._call, query, results, top_k)
        return [r["text"] for r in scored]

    async def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        """Retrieve and return results with Voyage relevance scores."""
        import asyncio

        results = await self._retriever.retrieve(
            query, top_k=self._fetch_k, metadata_filter=metadata_filter
        )
        if not results:
            return []
        return await asyncio.to_thread(self._call, query, results, top_k)
