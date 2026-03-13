"""Cross-Encoder Reranker: rerank retrieval results using a cross-encoder model."""

from __future__ import annotations

from .retriever import Retriever


class CrossEncoderReranker:
    """Rerank retrieval results using a cross-encoder model for higher precision.

    Cross-encoders score query-document pairs jointly, giving much more
    accurate relevance scores than bi-encoder (embedding) similarity alone.

    Usage::

        reranker = CrossEncoderReranker(
            retriever=retriever,
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        )
        results = await reranker.retrieve("What is RAG?", top_k=5)

    Requires ``sentence-transformers``: ``pip install synapsekit[semantic]``
    """

    def __init__(
        self,
        retriever: Retriever,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        fetch_k: int = 20,
    ) -> None:
        self._retriever = retriever
        self._model_name = model
        self._fetch_k = fetch_k
        self._cross_encoder = None

    def _get_cross_encoder(self):
        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise ImportError(
                    "sentence-transformers required for CrossEncoderReranker: "
                    "pip install synapsekit[semantic]"
                ) from None
            self._cross_encoder = CrossEncoder(self._model_name)
        return self._cross_encoder

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[str]:
        """Retrieve candidates then rerank with cross-encoder."""
        import asyncio

        # Get initial candidates from vector search
        results = await self._retriever.retrieve_with_scores(
            query, top_k=self._fetch_k, metadata_filter=metadata_filter
        )

        if not results:
            return []

        texts = [r["text"] for r in results]

        # Score query-document pairs with cross-encoder
        cross_encoder = self._get_cross_encoder()
        pairs = [[query, text] for text in texts]

        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(None, cross_encoder.predict, pairs)

        # Sort by cross-encoder score descending
        scored = sorted(zip(texts, scores, strict=False), key=lambda x: x[1], reverse=True)
        return [text for text, _score in scored[:top_k]]

    async def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        """Retrieve and return results with cross-encoder scores."""
        import asyncio

        results = await self._retriever.retrieve_with_scores(
            query, top_k=self._fetch_k, metadata_filter=metadata_filter
        )

        if not results:
            return []

        texts = [r["text"] for r in results]
        cross_encoder = self._get_cross_encoder()
        pairs = [[query, text] for text in texts]

        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(None, cross_encoder.predict, pairs)

        for result, score in zip(results, scores, strict=False):
            result["cross_encoder_score"] = float(score)

        results.sort(key=lambda r: r["cross_encoder_score"], reverse=True)
        return results[:top_k]
