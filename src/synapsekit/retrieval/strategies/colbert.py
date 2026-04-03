"""ColBERT retrieval strategy using late interaction (MaxSim)."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any


class ColBERTRetriever:
    """ColBERT retrieval with late interaction scoring (MaxSim).

    Uses RAGatouille under the hood to index documents with token embeddings
    and perform ColBERT-style late interaction during search.

    Usage::

        retriever = ColBERTRetriever(model="colbert-ir/colbertv2.0", index_name="docs")
        await retriever.add(["doc1", "doc2"], metadata=[{"id": 1}, {"id": 2}])
        results = await retriever.retrieve("my query", top_k=5)

    Requires ``ragatouille``: ``pip install synapsekit[colbert]``
    """

    def __init__(
        self,
        model: str = "colbert-ir/colbertv2.0",
        index_name: str = "colbert",
        index_root: str | None = None,
    ) -> None:
        self._model_name = model
        self._index_name = index_name
        self._index_root = index_root
        self._rag = None
        self._documents: list[str] = []
        self._metadata: list[dict] = []

    def _call_with_supported_kwargs(self, fn, /, *args, **kwargs):
        sig = inspect.signature(fn)
        supported = {k: v for k, v in kwargs.items() if v is not None and k in sig.parameters}
        return fn(*args, **supported)

    def _get_ragatouille(self):
        if self._rag is None:
            try:
                from ragatouille import RAGPretrainedModel
            except ImportError:
                raise ImportError(
                    "ragatouille required for ColBERTRetriever: pip install synapsekit[colbert]"
                ) from None

            self._rag = self._call_with_supported_kwargs(
                RAGPretrainedModel.from_pretrained,
                self._model_name,
                index_root=self._index_root,
            )
        return self._rag

    async def add(
        self,
        texts: list[str],
        metadata: list[dict] | None = None,
    ) -> None:
        """Index documents with ColBERT token embeddings."""
        if not texts:
            return

        if metadata is not None and len(metadata) != len(texts):
            raise ValueError("metadata length must match texts length")

        self._documents = list(texts)
        self._metadata = list(metadata) if metadata is not None else []

        rag = self._get_ragatouille()
        doc_ids = list(range(len(texts)))

        def _index():
            return self._call_with_supported_kwargs(
                rag.index,
                collection=texts,
                index_name=self._index_name,
                document_ids=doc_ids,
                index_root=self._index_root,
            )

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _index)

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[str]:
        """Search using ColBERT late interaction and return top text results."""
        results = await self.retrieve_with_scores(
            query, top_k=top_k, metadata_filter=metadata_filter
        )
        return [r["text"] for r in results]

    async def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        """Search and return results with scores (and metadata when available)."""
        _ = metadata_filter  # ColBERT backends typically do not support filtering yet.
        rag = self._get_ragatouille()

        def _search():
            return self._call_with_supported_kwargs(
                rag.search,
                query,
                k=top_k,
                top_k=top_k,
                index_name=self._index_name,
            )

        loop = asyncio.get_running_loop()
        raw_results = await loop.run_in_executor(None, _search)
        return self._normalize_results(raw_results, top_k=top_k)

    def _normalize_results(self, results: Any, top_k: int) -> list[dict]:
        if not results:
            return []

        normalized: list[dict] = []
        for item in results:
            text = None
            score = None
            doc_id = None

            if isinstance(item, dict):
                text = (
                    item.get("content")
                    or item.get("document")
                    or item.get("text")
                    or item.get("passage")
                )
                score = item.get("score")
                if "document_id" in item:
                    doc_id = item.get("document_id")
                elif "doc_id" in item:
                    doc_id = item.get("doc_id")
                elif "id" in item:
                    doc_id = item.get("id")
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                text = item[0]
                score = item[1]
            else:
                text = str(item)

            if text is None:
                continue

            entry: dict[str, Any] = {"text": text}
            if score is not None:
                entry["score"] = float(score)

            if doc_id is not None and self._metadata:
                try:
                    idx = int(doc_id)
                except (TypeError, ValueError):
                    idx = None
                if idx is not None and 0 <= idx < len(self._metadata):
                    entry["metadata"] = self._metadata[idx]

            normalized.append(entry)
            if len(normalized) >= top_k:
                break

        return normalized
