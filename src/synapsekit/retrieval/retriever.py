from __future__ import annotations

from .base import VectorStore


class Retriever:
    """
    Retrieves relevant chunks from a vectorstore.
    Optionally re-ranks results using BM25 (rank-bm25).
    """

    def __init__(
        self,
        vectorstore: VectorStore,
        rerank: bool = False,
    ) -> None:
        self._store = vectorstore
        self._rerank = rerank

    async def add(
        self,
        texts: list[str],
        metadata: list[dict] | None = None,
    ) -> None:
        """Add texts to the underlying vector store."""
        await self._store.add(texts, metadata)

    def _bm25_rerank(self, query: str, texts: list[str], top_k: int) -> list[str]:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank-bm25 required: pip install rank-bm25") from None
        tokenized = [t.lower().split() for t in texts]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(query.lower().split())
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [texts[i] for i in ranked_indices[:top_k]]

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[str]:
        """Return top_k relevant text chunks for query."""
        from ..observe.runtime import end_span, record_exception, start_span

        fetch_k = top_k * 3 if self._rerank else top_k
        results = await self._store.search(query, top_k=fetch_k, metadata_filter=metadata_filter)

        if not results:
            return []

        texts = [r["text"] for r in results]

        if self._rerank and len(texts) > 1:
            rerank_span = start_span(
                "reranker.rerank",
                {
                    "reranker.type": "bm25",
                    "reranker.top_k": top_k,
                    "reranker.candidates": len(texts),
                },
            )
            try:
                reranked = self._bm25_rerank(query, texts, top_k)
                end_span(rerank_span, attributes={"rag.retrieved_chunks": len(reranked)})
                return reranked
            except Exception as exc:
                record_exception(rerank_span, exc)
                end_span(rerank_span, error=exc)
                raise

        return texts[:top_k]

    async def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        """Return top_k results with scores and metadata."""
        from ..observe.runtime import end_span, record_exception, start_span

        fetch_k = top_k * 3 if self._rerank else top_k
        results = await self._store.search(query, top_k=fetch_k, metadata_filter=metadata_filter)

        if not results or not self._rerank:
            return results[:top_k]

        texts = [r["text"] for r in results]
        rerank_span = start_span(
            "reranker.rerank",
            {
                "reranker.type": "bm25",
                "reranker.top_k": top_k,
                "reranker.candidates": len(texts),
            },
        )
        try:
            reranked = self._bm25_rerank(query, texts, top_k)
            text_to_result = {r["text"]: r for r in results}
            resolved = [text_to_result[t] for t in reranked if t in text_to_result]
            end_span(rerank_span, attributes={"rag.retrieved_chunks": len(resolved)})
            return resolved
        except Exception as exc:
            record_exception(rerank_span, exc)
            end_span(rerank_span, error=exc)
            raise

    async def retrieve_mmr(
        self,
        query: str,
        top_k: int = 5,
        lambda_mult: float = 0.5,
        fetch_k: int = 20,
        metadata_filter: dict | None = None,
    ) -> list[str]:
        """Retrieve using Maximal Marginal Relevance for diversity."""
        results = await self._store.search_mmr(
            query,
            top_k=top_k,
            lambda_mult=lambda_mult,
            fetch_k=fetch_k,
            metadata_filter=metadata_filter,
        )
        return [r["text"] for r in results]
