"""Ensemble Retrieval: fuse results from multiple retrievers."""

from __future__ import annotations

import asyncio

from .retriever import Retriever


class EnsembleRetriever:
    """Ensemble Retrieval: combines results from multiple retrievers using
    Reciprocal Rank Fusion (RRF) for better recall and diversity.

    All retrievers are queried **concurrently** via ``asyncio.gather``, so
    total latency is bounded by the slowest individual retriever rather than
    the sum of all retriever latencies.

    Usage::

        ensemble = EnsembleRetriever(
            retrievers=[retriever_a, retriever_b],
            weights=[0.7, 0.3],
        )
        results = await ensemble.retrieve("What is RAG?", top_k=5)
    """

    def __init__(
        self,
        retrievers: list[Retriever],
        weights: list[float] | None = None,
        rrf_k: int = 60,
    ) -> None:
        if not retrievers:
            raise ValueError("At least one retriever is required.")
        if weights is not None and len(weights) != len(retrievers):
            raise ValueError("weights must have the same length as retrievers.")
        self._retrievers = retrievers
        self._weights = weights or [1.0] * len(retrievers)
        self._rrf_k = rrf_k

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[str]:
        """Retrieve from all retrievers concurrently and fuse with weighted RRF."""
        # Fan out to all retrievers in parallel — no sequential waiting
        all_results: list[list[str]] = await asyncio.gather(
            *[
                retriever.retrieve(query, top_k=top_k, metadata_filter=metadata_filter)
                for retriever in self._retrievers
            ]
        )

        scores: dict[str, float] = {}
        for results, weight in zip(all_results, self._weights, strict=True):
            for rank, text in enumerate(results):
                scores[text] = scores.get(text, 0.0) + weight / (self._rrf_k + rank + 1)

        sorted_texts = sorted(scores, key=lambda t: scores[t], reverse=True)
        return sorted_texts[:top_k]
