"""Retriever wrapper that uses full-context ingestion when documents fit token budget."""

from __future__ import annotations

from ..text_splitters.base import BaseSplitter
from ..text_splitters.recursive import RecursiveCharacterTextSplitter
from .retriever import Retriever
from .token_counting import TokenCounter


class FullContextRetriever:
    """Auto-switch between full-document and chunked ingestion."""

    def __init__(
        self,
        retriever: Retriever,
        max_doc_tokens: int = 100_000,
        token_counter: TokenCounter | None = None,
        model: str | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        splitter: BaseSplitter | None = None,
    ) -> None:
        if max_doc_tokens <= 0:
            raise ValueError("max_doc_tokens must be > 0")

        self._retriever = retriever
        self._max_doc_tokens = max_doc_tokens
        self._counter = token_counter or TokenCounter(model=model, backend="auto")
        self._splitter: BaseSplitter = splitter or RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    async def add_document(self, text: str, metadata: dict | None = None) -> None:
        text = text.strip()
        if not text:
            return

        doc_tokens = self._counter.count_cached(text)
        base_meta = dict(metadata or {})

        if doc_tokens <= self._max_doc_tokens:
            meta = {**base_meta, "full_context": True, "doc_tokens": doc_tokens}
            await self._retriever.add([text], [meta])
            return

        chunks = self._splitter.split_with_metadata(text, base_meta)
        chunk_texts = [c["text"] for c in chunks]
        chunk_meta = [
            {
                **c["metadata"],
                "full_context": False,
                "doc_tokens": doc_tokens,
                "chunk_tokens": self._counter.count_cached(c["text"]),
            }
            for c in chunks
        ]

        await self._retriever.add(chunk_texts, chunk_meta)

    async def add(self, texts: list[str], metadata: list[dict] | None = None) -> None:
        if not texts:
            return
        meta_list = metadata or [{} for _ in texts]
        for text, meta in zip(texts, meta_list, strict=False):
            await self.add_document(text, metadata=meta)

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[str]:
        return await self._retriever.retrieve(query, top_k=top_k, metadata_filter=metadata_filter)

    async def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        if hasattr(self._retriever, "retrieve_with_scores"):
            return await self._retriever.retrieve_with_scores(  # type: ignore[return-value]
                query,
                top_k=top_k,
                metadata_filter=metadata_filter,
            )

        texts = await self.retrieve(query, top_k=top_k, metadata_filter=metadata_filter)
        return [{"text": text, "score": None, "metadata": {}} for text in texts]
