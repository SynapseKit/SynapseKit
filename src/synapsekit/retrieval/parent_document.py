"""Parent Document Retrieval: embed small chunks, return the full parent document."""

from __future__ import annotations

import uuid

from .retriever import Retriever


class ParentDocumentRetriever:
    """Parent Document Retrieval: splits documents into small chunks for
    fine-grained embedding, but returns the full parent document at retrieval time.

    This gives the best of both worlds: precise search on small chunks,
    but rich context from the full document.

    Usage::

        pdr = ParentDocumentRetriever(retriever=retriever, chunk_size=200)
        await pdr.add_documents(["Full document one...", "Full document two..."])
        results = await pdr.retrieve("query", top_k=3)
        # Returns full parent documents, not small chunks
    """

    def __init__(
        self,
        retriever: Retriever,
        chunk_size: int = 200,
        chunk_overlap: int = 50,
    ) -> None:
        self._retriever = retriever
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._parents: dict[str, str] = {}  # parent_id -> full text

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        if len(text) <= self._chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + self._chunk_size
            chunks.append(text[start:end])
            start = end - self._chunk_overlap
            if start >= len(text):
                break
        return chunks

    async def add_documents(
        self,
        texts: list[str],
        metadata: list[dict] | None = None,
    ) -> None:
        """Split documents into chunks, embed them, and store parent references."""
        base_meta = metadata or [{} for _ in texts]
        all_chunks: list[str] = []
        all_metadata: list[dict] = []

        for doc_idx, text in enumerate(texts):
            if not text or not text.strip():
                continue

            parent_id = str(uuid.uuid4())
            self._parents[parent_id] = text

            chunks = self._chunk_text(text)
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadata.append(
                    {
                        **base_meta[doc_idx],
                        "_parent_id": parent_id,
                    }
                )

        if all_chunks:
            await self._retriever.add(all_chunks, all_metadata)

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[str]:
        """Retrieve matching chunks, then return their full parent documents."""
        # Fetch more chunks than needed since multiple chunks may map to same parent
        results = await self._retriever.retrieve_with_scores(
            query, top_k=top_k * 3, metadata_filter=metadata_filter
        )

        seen_parents: set[str] = set()
        parents: list[str] = []

        for result in results:
            parent_id = result.get("metadata", {}).get("_parent_id")
            if parent_id and parent_id not in seen_parents and parent_id in self._parents:
                seen_parents.add(parent_id)
                parents.append(self._parents[parent_id])
                if len(parents) >= top_k:
                    break

        return parents
