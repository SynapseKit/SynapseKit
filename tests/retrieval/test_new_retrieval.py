"""Tests for RAPTORRetriever, AgenticRAGRetriever, DocumentAugmentationRetriever,
LateChunkingRetriever."""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock

import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc(text: str, metadata=None):
    from synapsekit.loaders.base import Document

    return Document(text=text, metadata=metadata or {})


def _make_retriever(results=None):
    r = AsyncMock()
    r.retrieve = AsyncMock(return_value=results or [])
    return r


def _make_llm(response: str = "summary"):
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=response)
    return llm


# ---------------------------------------------------------------------------
# RAPTORRetriever
# ---------------------------------------------------------------------------


class TestRAPTORRetriever:
    @pytest.mark.asyncio
    async def test_retrieve_returns_documents(self):
        from synapsekit.retrieval.raptor import RAPTORRetriever

        docs = ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6"]
        retriever = _make_retriever(docs)
        llm = _make_llm("summary text")
        raptor = RAPTORRetriever(base_retriever=retriever, llm=llm, levels=1, cluster_size=3)
        results = await raptor.retrieve("query", top_k=3)
        assert isinstance(results, list)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_fallback_when_few_docs(self):
        from synapsekit.retrieval.raptor import RAPTORRetriever

        docs = ["doc1", "doc2"]
        retriever = _make_retriever(docs)
        llm = _make_llm()
        raptor = RAPTORRetriever(base_retriever=retriever, llm=llm, cluster_size=5)
        results = await raptor.retrieve("query", top_k=5)
        # Falls back to base results
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_deduplication(self):
        from synapsekit.retrieval.raptor import RAPTORRetriever

        # Retriever always returns same docs
        docs = ["doc_a", "doc_b", "doc_c", "doc_d", "doc_e"]
        retriever = _make_retriever(docs)
        llm = _make_llm("summary")
        raptor = RAPTORRetriever(base_retriever=retriever, llm=llm, levels=1, cluster_size=3)
        results = await raptor.retrieve("query", top_k=10)
        texts = [d.text for d in results]
        assert len(texts) == len(set(texts)), "duplicate texts found"

    @pytest.mark.asyncio
    async def test_accepts_document_objects(self):
        from synapsekit.retrieval.raptor import RAPTORRetriever

        docs = [_make_doc(f"doc {i}") for i in range(6)]
        retriever = _make_retriever(docs)
        llm = _make_llm("summary")
        raptor = RAPTORRetriever(base_retriever=retriever, llm=llm, levels=1, cluster_size=3)
        results = await raptor.retrieve("query", top_k=3)
        assert all(hasattr(d, "text") for d in results)

    def test_retrieve_is_coroutine(self):
        from synapsekit.retrieval.raptor import RAPTORRetriever

        assert inspect.iscoroutinefunction(RAPTORRetriever.retrieve)


# ---------------------------------------------------------------------------
# AgenticRAGRetriever
# ---------------------------------------------------------------------------


class TestAgenticRAGRetriever:
    @pytest.mark.asyncio
    async def test_search_then_done(self):
        from synapsekit.retrieval.agentic_rag import AgenticRAGRetriever

        docs = ["result1", "result2"]
        retriever = _make_retriever(docs)
        llm = _make_llm("SEARCH: what is quantum computing")
        # Second call returns DONE
        llm.generate.side_effect = [
            "SEARCH: what is quantum computing",
            "DONE",
        ]
        agent = AgenticRAGRetriever(retriever=retriever, llm=llm, max_iterations=3)
        results = await agent.retrieve("What is quantum computing?", top_k=5)
        assert isinstance(results, list)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_done_immediately_falls_back_to_direct_retrieval(self):
        from synapsekit.retrieval.agentic_rag import AgenticRAGRetriever

        docs = ["doc1"]
        retriever = _make_retriever(docs)
        llm = _make_llm("DONE")
        agent = AgenticRAGRetriever(retriever=retriever, llm=llm, max_iterations=2)
        results = await agent.retrieve("query", top_k=5)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_max_iterations_respected(self):
        from synapsekit.retrieval.agentic_rag import AgenticRAGRetriever

        retriever = _make_retriever(["doc"])
        llm = _make_llm("SEARCH: more info")
        agent = AgenticRAGRetriever(retriever=retriever, llm=llm, max_iterations=2)
        await agent.retrieve("query", top_k=5)
        # LLM called at most max_iterations times
        assert llm.generate.call_count <= 2

    @pytest.mark.asyncio
    async def test_deduplication(self):
        from synapsekit.retrieval.agentic_rag import AgenticRAGRetriever

        retriever = _make_retriever(["doc_a", "doc_b"])
        llm = AsyncMock()
        llm.generate.side_effect = [
            "SEARCH: first query",
            "SEARCH: second query",
            "DONE",
        ]
        agent = AgenticRAGRetriever(retriever=retriever, llm=llm, max_iterations=3)
        results = await agent.retrieve("query", top_k=10)
        texts = [d.text for d in results]
        assert len(texts) == len(set(texts))

    def test_retrieve_is_coroutine(self):
        from synapsekit.retrieval.agentic_rag import AgenticRAGRetriever

        assert inspect.iscoroutinefunction(AgenticRAGRetriever.retrieve)


# ---------------------------------------------------------------------------
# DocumentAugmentationRetriever
# ---------------------------------------------------------------------------


class TestDocumentAugmentationRetriever:
    @pytest.mark.asyncio
    async def test_expand_queries(self):
        from synapsekit.retrieval.document_augmentation import DocumentAugmentationRetriever

        retriever = _make_retriever(["doc1", "doc2"])
        llm = _make_llm("alt query 1\nalt query 2\nalt query 3")
        aug = DocumentAugmentationRetriever(
            base_retriever=retriever,
            llm=llm,
            expand_queries=True,
            expand_documents=False,
            n_queries=3,
        )
        results = await aug.retrieve("original query", top_k=5)
        assert isinstance(results, list)
        # Multiple queries were issued
        assert retriever.retrieve.call_count >= 2

    @pytest.mark.asyncio
    async def test_expand_documents(self):
        from synapsekit.retrieval.document_augmentation import DocumentAugmentationRetriever

        retriever = _make_retriever(["doc1"])
        llm = _make_llm("hypothetical answer")
        aug = DocumentAugmentationRetriever(
            base_retriever=retriever,
            llm=llm,
            expand_queries=False,
            expand_documents=True,
        )
        results = await aug.retrieve("query", top_k=5)
        assert len(results) == 1
        assert "hypothetical answer" in results[0].text

    @pytest.mark.asyncio
    async def test_deduplication_across_queries(self):
        from synapsekit.retrieval.document_augmentation import DocumentAugmentationRetriever

        retriever = _make_retriever(["same_doc"])
        llm = AsyncMock()
        # query expansion returns 2 alternatives; doc expansion returns short answer
        llm.generate.side_effect = [
            "alt1\nalt2",  # query expansion
            "answer",  # doc expansion
        ]
        aug = DocumentAugmentationRetriever(
            base_retriever=retriever,
            llm=llm,
            expand_queries=True,
            expand_documents=True,
            n_queries=2,
        )
        results = await aug.retrieve("query", top_k=10)
        # same_doc appears once despite multiple queries
        base_texts = [r.text.split("\n\n")[0] for r in results]
        assert len([t for t in base_texts if "same_doc" in t]) == 1

    @pytest.mark.asyncio
    async def test_no_expansion(self):
        from synapsekit.retrieval.document_augmentation import DocumentAugmentationRetriever

        retriever = _make_retriever(["doc1", "doc2"])
        llm = _make_llm()
        aug = DocumentAugmentationRetriever(
            base_retriever=retriever,
            llm=llm,
            expand_queries=False,
            expand_documents=False,
        )
        results = await aug.retrieve("query", top_k=5)
        assert len(results) == 2
        llm.generate.assert_not_called()

    def test_retrieve_is_coroutine(self):
        from synapsekit.retrieval.document_augmentation import DocumentAugmentationRetriever

        assert inspect.iscoroutinefunction(DocumentAugmentationRetriever.retrieve)


# ---------------------------------------------------------------------------
# LateChunkingRetriever
# ---------------------------------------------------------------------------


class TestLateChunkingRetriever:
    def _make_embeddings(self, dim: int = 4):
        emb = AsyncMock()

        async def embed(texts):
            return np.ones((len(texts), dim), dtype=np.float32) / np.sqrt(dim)

        async def embed_one(text):
            return np.ones(dim, dtype=np.float32) / np.sqrt(dim)

        emb.embed = embed
        emb.embed_one = embed_one
        return emb

    @pytest.mark.asyncio
    async def test_retrieve_returns_documents(self):
        from synapsekit.retrieval.late_chunking import LateChunkingRetriever

        emb = self._make_embeddings()
        texts = ["This is a long document with several words that can be chunked."]
        retriever = LateChunkingRetriever(texts=texts, embedding_backend=emb, chunk_size=20, chunk_overlap=5)
        results = await retriever.retrieve("query", top_k=3)
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(hasattr(r, "text") for r in results)

    @pytest.mark.asyncio
    async def test_lazy_init_only_once(self):
        from unittest.mock import AsyncMock as _AsyncMock

        import numpy as np

        from synapsekit.retrieval.late_chunking import LateChunkingRetriever

        dim = 4
        emb = _AsyncMock()
        emb.embed = _AsyncMock(
            return_value=np.ones((2, dim), dtype=np.float32) / np.sqrt(dim)
        )
        emb.embed_one = _AsyncMock(
            return_value=np.ones(dim, dtype=np.float32) / np.sqrt(dim)
        )

        texts = ["document one", "document two"]
        retriever = LateChunkingRetriever(texts=texts, embedding_backend=emb, chunk_size=5, chunk_overlap=1)

        await retriever.retrieve("query")
        call_count_after_first = emb.embed.call_count
        await retriever.retrieve("query")
        # embed should not be called again on second retrieve (lazy init)
        assert emb.embed.call_count == call_count_after_first

    @pytest.mark.asyncio
    async def test_empty_texts(self):
        from synapsekit.retrieval.late_chunking import LateChunkingRetriever

        emb = self._make_embeddings()
        retriever = LateChunkingRetriever(texts=[], embedding_backend=emb)
        results = await retriever.retrieve("query")
        assert results == []

    @pytest.mark.asyncio
    async def test_top_k_respected(self):
        from synapsekit.retrieval.late_chunking import LateChunkingRetriever

        emb = self._make_embeddings()
        texts = ["a " * 200]  # Long enough to produce many chunks
        retriever = LateChunkingRetriever(texts=texts, embedding_backend=emb, chunk_size=10, chunk_overlap=2)
        results = await retriever.retrieve("query", top_k=2)
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_chunk_metadata_present(self):
        from synapsekit.retrieval.late_chunking import LateChunkingRetriever

        emb = self._make_embeddings()
        texts = ["hello world"]
        retriever = LateChunkingRetriever(texts=texts, embedding_backend=emb, chunk_size=5, chunk_overlap=0)
        results = await retriever.retrieve("hello", top_k=5)
        for r in results:
            assert "doc_idx" in r.metadata
            assert "chunk_idx" in r.metadata

    def test_retrieve_is_coroutine(self):
        from synapsekit.retrieval.late_chunking import LateChunkingRetriever

        assert inspect.iscoroutinefunction(LateChunkingRetriever.retrieve)
