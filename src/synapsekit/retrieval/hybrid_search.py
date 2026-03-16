"""Hybrid Search Retriever: combines BM25 keyword matching with vector similarity."""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from .retriever import Retriever


class HybridSearchRetriever:
    """Combines BM25 keyword matching with vector similarity using RRF fusion.

    Usage::

        hybrid = HybridSearchRetriever(retriever=retriever)
        hybrid.add_documents(["doc1 text", "doc2 text", ...])
        results = await hybrid.retrieve("search query", top_k=5)
    """

    def __init__(
        self,
        retriever: Retriever,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
        rrf_k: int = 60,
    ) -> None:
        self._retriever = retriever
        self._bm25_weight = bm25_weight
        self._vector_weight = vector_weight
        self._rrf_k = rrf_k
        self._documents: list[str] = []
        self._bm25: BM25Okapi | None = None

    def add_documents(self, texts: list[str]) -> None:
        """Build the BM25 index from the given texts."""
        self._documents = list(texts)
        tokenized = [doc.lower().split() for doc in texts]
        self._bm25 = BM25Okapi(tokenized)

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[str]:
        """Retrieve using RRF fusion of BM25 + vector scores."""
        # Vector retrieval
        vector_results = await self._retriever.retrieve(
            query, top_k=top_k * 2, metadata_filter=metadata_filter
        )

        # BM25 scoring
        bm25_ranked: list[str] = []
        if self._bm25 is not None and self._documents:
            scores = self._bm25.get_scores(query.lower().split())
            scored_pairs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            bm25_ranked = [self._documents[i] for i, _ in scored_pairs[: top_k * 2]]

        # RRF fusion
        fused_scores: dict[str, float] = {}

        for rank, doc in enumerate(vector_results):
            fused_scores[doc] = fused_scores.get(doc, 0.0) + self._vector_weight / (
                self._rrf_k + rank + 1
            )

        for rank, doc in enumerate(bm25_ranked):
            fused_scores[doc] = fused_scores.get(doc, 0.0) + self._bm25_weight / (
                self._rrf_k + rank + 1
            )

        sorted_docs = sorted(fused_scores, key=fused_scores.__getitem__, reverse=True)
        return sorted_docs[:top_k]
