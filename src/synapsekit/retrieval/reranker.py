"""Base contract for all SynapseKit rerankers.

Every reranker — hosted (Cohere, Voyage, Jina, mixedbread) and local
(cross-encoder) — implements the same uniform async interface:

- ``retrieve(query, top_k)`` -> ``list[str]``
- ``retrieve_with_scores(query, top_k)`` -> ``list[dict]``
  with ``{"text": ..., "relevance_score": ...}``

Rerankers wrap a ``Retriever`` (or anything exposing the same
``retrieve``/``retrieve_with_scores`` async methods): they fetch ``fetch_k``
candidates, score them with the provider model, and return the top ``top_k``
in relevance order.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .retriever import Retriever


class Reranker(ABC):
    """Abstract base class for all reranker implementations."""

    def __init__(
        self,
        retriever: Retriever,
        model: str,
        api_key: str | None = None,
        fetch_k: int = 20,
    ) -> None:
        self._retriever = retriever
        self._model = model
        self._api_key = api_key
        self._fetch_k = fetch_k

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[str]:
        """Return the top-``top_k`` reranked document texts."""
        raise NotImplementedError

    @abstractmethod
    async def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        """Return the top-``top_k`` reranked results with scores."""
        raise NotImplementedError
