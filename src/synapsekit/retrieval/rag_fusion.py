from __future__ import annotations

from ..llm.base import BaseLLM
from .retriever import Retriever


class RAGFusionRetriever:
    """RAG Fusion: multi-query retrieval with Reciprocal Rank Fusion.

    Generates multiple query variations using an LLM, retrieves results for
    each, and fuses them using RRF scoring for better recall.

    Usage::

        fusion = RAGFusionRetriever(retriever=retriever, llm=llm)
        results = await fusion.retrieve("What is quantum computing?", top_k=5)
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: BaseLLM,
        num_queries: int = 3,
        rrf_k: int = 60,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._num_queries = num_queries
        self._rrf_k = rrf_k

    async def _generate_queries(self, query: str) -> list[str]:
        """Use the LLM to generate query variations."""
        prompt = (
            f"Generate {self._num_queries} different search queries related to "
            f"the following question. Return only the queries, one per line, "
            f"without numbering or bullets.\n\n"
            f"Question: {query}"
        )
        response = await self._llm.generate(prompt)
        queries = [q.strip() for q in response.strip().split("\n") if q.strip()]
        # Always include the original query
        return [query, *queries[: self._num_queries]]

    def _reciprocal_rank_fusion(
        self,
        result_lists: list[list[str]],
    ) -> list[str]:
        """Fuse multiple ranked lists using Reciprocal Rank Fusion."""
        scores: dict[str, float] = {}
        for results in result_lists:
            for rank, text in enumerate(results):
                scores[text] = scores.get(text, 0.0) + 1.0 / (self._rrf_k + rank + 1)

        sorted_texts = sorted(scores, key=lambda t: scores[t], reverse=True)
        return sorted_texts

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[str]:
        """Retrieve with RAG Fusion: multi-query + RRF."""
        queries = await self._generate_queries(query)

        all_results: list[list[str]] = []
        for q in queries:
            results = await self._retriever.retrieve(
                q, top_k=top_k, metadata_filter=metadata_filter
            )
            all_results.append(results)

        fused = self._reciprocal_rank_fusion(all_results)
        return fused[:top_k]
