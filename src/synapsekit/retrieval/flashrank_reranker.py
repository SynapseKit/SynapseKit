"""FlashRank local CPU reranker: ultra-fast, zero-GPU, zero-API-key reranking."""

from __future__ import annotations

import asyncio
from typing import Any

from .reranker import Reranker
from .retriever import Retriever


class FlashRankReranker(Reranker):
    """Rerank retrieval results locally on CPU using FlashRank.

    FlashRank uses quantized, ultra-compact ONNX models (~4MB to ~40MB) to score
    and rerank candidate passages locally with low latency (<15ms) without requiring
    PyTorch, CUDA, or paid external API keys.

    Usage::

        from synapsekit.retrieval import FlashRankReranker

        reranker = FlashRankReranker(
            retriever=retriever,
            model="ms-marco-TinyBERT-L-2-v2",
            fetch_k=20,
        )
        results = await reranker.retrieve("What is RAG?", top_k=5)
        scored_results = await reranker.retrieve_with_scores("What is RAG?", top_k=5)

    Requires ``flashrank``: ``pip install synapsekit[flashrank]`` or ``pip install flashrank``.
    """

    DEFAULT_MODEL = "ms-marco-TinyBERT-L-2-v2"

    def __init__(
        self,
        retriever: Retriever,
        model: str = DEFAULT_MODEL,
        fetch_k: int = 20,
        cache_dir: str | None = None,
        max_length: int = 512,
    ) -> None:
        super().__init__(retriever, model, fetch_k=fetch_k)
        self._cache_dir = cache_dir
        self._max_length = max_length
        self._ranker: Any = None

    def _get_ranker(self) -> Any:
        if self._ranker is None:
            try:
                from flashrank import Ranker
            except ImportError:
                raise ImportError(
                    "flashrank is required for FlashRankReranker: "
                    "pip install synapsekit[flashrank] or pip install flashrank"
                ) from None
            self._ranker = Ranker(
                model_name=self._model,
                cache_dir=self._cache_dir,
                max_length=self._max_length,
            )
        return self._ranker

    def _call(self, query: str, documents: list[Any], top_n: int) -> list[dict]:
        from flashrank import RerankRequest

        ranker = self._get_ranker()

        passages: list[dict] = []
        for idx, doc in enumerate(documents):
            if isinstance(doc, dict):
                item = dict(doc)
                item.setdefault("id", idx)
                passages.append(item)
            else:
                passages.append({"id": idx, "text": str(doc)})

        rerank_req = RerankRequest(query=query, passages=passages)
        raw_results = ranker.rerank(rerank_req)

        scored: list[dict] = []
        for r in raw_results:
            score = float(r.get("score", 0.0))
            entry: dict[str, Any] = {"text": r.get("text", "")}
            for k, v in r.items():
                if k not in ("text", "score", "id", "relevance_score"):
                    entry[k] = v
            if "relevance_score" in r and "initial_score" not in entry:
                entry["initial_score"] = r["relevance_score"]
            entry["relevance_score"] = score
            scored.append(entry)

        scored.sort(key=lambda x: x["relevance_score"], reverse=True)
        return scored[:top_n]

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[str]:
        """Retrieve initial candidates then rerank using FlashRank locally on CPU."""
        results: list[Any]
        if hasattr(self._retriever, "retrieve_with_scores"):
            results = await self._retriever.retrieve_with_scores(
                query, top_k=self._fetch_k, metadata_filter=metadata_filter
            )
        else:
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
        """Retrieve and return top candidates with FlashRank relevance scores."""
        results: list[Any]
        if hasattr(self._retriever, "retrieve_with_scores"):
            results = await self._retriever.retrieve_with_scores(
                query, top_k=self._fetch_k, metadata_filter=metadata_filter
            )
        else:
            results = await self._retriever.retrieve(
                query, top_k=self._fetch_k, metadata_filter=metadata_filter
            )

        if not results:
            return []

        return await asyncio.to_thread(self._call, query, results, top_k)
