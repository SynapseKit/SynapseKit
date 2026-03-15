"""Cohere Reranker: rerank retrieval results using the Cohere Rerank API."""

from __future__ import annotations

import os

from .retriever import Retriever


class CohereReranker:
    """Rerank retrieval results using Cohere's rerank models.

    Usage::

        reranker = CohereReranker(retriever=retriever, model="rerank-v3.5")
        results = await reranker.retrieve("What is RAG?", top_k=5)

    Requires ``cohere``: ``pip install synapsekit[cohere]``
    """

    def __init__(
        self,
        retriever: Retriever,
        model: str = "rerank-v3.5",
        api_key: str | None = None,
        fetch_k: int = 20,
    ) -> None:
        self._retriever = retriever
        self._model = model
        self._api_key = api_key
        self._fetch_k = fetch_k
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from cohere import ClientV2
            except ImportError:
                raise ImportError(
                    "cohere required for CohereReranker: pip install synapsekit[cohere]"
                ) from None
            key = self._api_key or os.environ.get("CO_API_KEY", "")
            self._client = ClientV2(api_key=key)
        return self._client

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[str]:
        """Retrieve candidates then rerank with the Cohere Rerank API."""
        results = await self._retriever.retrieve(
            query, top_k=self._fetch_k, metadata_filter=metadata_filter
        )

        if not results:
            return []

        client = self._get_client()
        reranked = client.rerank(
            model=self._model,
            query=query,
            documents=results,
            top_n=top_k,
        )

        return [results[r.index] for r in reranked.results]

    async def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        """Retrieve and return results with Cohere relevance scores."""
        results = await self._retriever.retrieve(
            query, top_k=self._fetch_k, metadata_filter=metadata_filter
        )

        if not results:
            return []

        client = self._get_client()
        reranked = client.rerank(
            model=self._model,
            query=query,
            documents=results,
            top_n=top_k,
        )

        return [
            {"text": results[r.index], "relevance_score": r.relevance_score}
            for r in reranked.results
        ]
