from __future__ import annotations

import re

from .retriever import Retriever


class SentenceWindowRetriever:
    """Sentence Window Retrieval: embed individual sentences, return surrounding window.

    Splits documents into sentences for fine-grained embedding, but returns a
    window of surrounding sentences for richer context at retrieval time.

    Usage::

        swr = SentenceWindowRetriever(retriever=retriever, window_size=2)
        await swr.add_documents(["Full document text here..."])
        results = await swr.retrieve("query", top_k=3)
    """

    def __init__(
        self,
        retriever: Retriever,
        window_size: int = 2,
    ) -> None:
        self._retriever = retriever
        self._window_size = window_size
        # Store original sentences for window expansion
        self._doc_sentences: list[list[str]] = []

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    async def add_documents(
        self,
        texts: list[str],
        metadata: list[dict] | None = None,
    ) -> None:
        """Split texts into sentences, embed each, and store for window retrieval."""
        all_sentences: list[str] = []
        all_metadata: list[dict] = []
        base_meta = metadata or [{} for _ in texts]

        for doc_idx, text in enumerate(texts):
            sentences = self._split_sentences(text)
            if not sentences:
                continue

            self._doc_sentences.append(sentences)
            doc_ref = len(self._doc_sentences) - 1

            for sent_idx, sentence in enumerate(sentences):
                all_sentences.append(sentence)
                all_metadata.append(
                    {
                        **base_meta[doc_idx],
                        "_sw_doc": doc_ref,
                        "_sw_sent": sent_idx,
                    }
                )

        if all_sentences:
            await self._retriever.add(all_sentences, all_metadata)

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[str]:
        """Retrieve sentences and expand to surrounding window."""
        results = await self._retriever.retrieve_with_scores(
            query, top_k=top_k, metadata_filter=metadata_filter
        )

        expanded: list[str] = []
        seen: set[tuple[int, int]] = set()

        for result in results:
            meta = result.get("metadata", {})
            doc_ref = meta.get("_sw_doc")
            sent_idx = meta.get("_sw_sent")

            if doc_ref is None or sent_idx is None:
                # Not a sentence-window chunk, return as-is
                expanded.append(result["text"])
                continue

            key = (doc_ref, sent_idx)
            if key in seen:
                continue
            seen.add(key)

            sentences = self._doc_sentences[doc_ref]
            start = max(0, sent_idx - self._window_size)
            end = min(len(sentences), sent_idx + self._window_size + 1)
            window = " ".join(sentences[start:end])
            expanded.append(window)

        return expanded
