"""LateChunkingRetriever: late chunking retrieval (chunk after full-doc embedding)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..embeddings.backend import SynapsekitEmbeddings
    from ..loaders.base import Document


class LateChunkingRetriever:
    """Late chunking retrieval.

    Unlike standard chunking (chunk → embed each), late chunking embeds the
    full document first then splits into chunks, assigning each chunk the
    full-document embedding as an approximation of contextual embedding.

    Chunks are indexed lazily on the first ``retrieve()`` call.

    Usage::

        retriever = LateChunkingRetriever(
            texts=["Long document..."],
            embedding_backend=embeddings,
        )
        results = await retriever.retrieve("query", top_k=5)
    """

    def __init__(
        self,
        texts: list[str],
        embedding_backend: SynapsekitEmbeddings,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        self._texts = texts
        self._embeddings = embedding_backend
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

        # Lazy-init state
        self._chunks: list[str] = []
        self._chunk_meta: list[dict] = []
        self._vectors: np.ndarray | None = None  # (N, D)
        self._initialised = False

    def _split_into_chunks(self, text: str, doc_idx: int) -> list[tuple[str, dict]]:
        """Split text into character-based chunks with overlap."""
        chunks: list[tuple[str, dict]] = []
        step = max(1, self._chunk_size - self._chunk_overlap)
        offset = 0
        chunk_idx = 0
        while offset < len(text):
            end = min(offset + self._chunk_size, len(text))
            chunk_text = text[offset:end]
            chunks.append(
                (
                    chunk_text,
                    {"doc_idx": doc_idx, "chunk_idx": chunk_idx, "offset": offset},
                )
            )
            offset += step
            chunk_idx += 1
        return chunks

    async def _init_index(self) -> None:
        """Embed full documents and build the chunk index."""
        if self._initialised:
            return

        if not self._texts:
            self._initialised = True
            return

        # Embed full documents
        doc_vectors = await self._embeddings.embed(self._texts)  # (D_docs, D)

        all_chunks: list[str] = []
        all_meta: list[dict] = []
        chunk_doc_indices: list[int] = []

        for doc_idx, text in enumerate(self._texts):
            for chunk_text, meta in self._split_into_chunks(text, doc_idx):
                all_chunks.append(chunk_text)
                all_meta.append(meta)
                chunk_doc_indices.append(doc_idx)

        # Assign each chunk the full-document embedding (contextual approximation)
        if all_chunks:
            vectors = doc_vectors[chunk_doc_indices]  # (N_chunks, D)
            self._vectors = vectors
        else:
            self._vectors = None

        self._chunks = all_chunks
        self._chunk_meta = all_meta
        self._initialised = True

    async def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Retrieve top_k chunks most similar to query."""
        from ..loaders.base import Document

        await self._init_index()

        if not self._chunks or self._vectors is None:
            return []

        q_vec = await self._embeddings.embed_one(query)  # (D,)
        scores = self._vectors @ q_vec  # (N,)

        k = min(top_k, len(self._chunks))
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        return [
            Document(
                text=self._chunks[i],
                metadata={**self._chunk_meta[i], "score": float(scores[i])},
            )
            for i in top_idx
        ]
