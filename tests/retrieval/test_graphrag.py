from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from synapsekit.retrieval.graphrag import GraphRAGRetriever, KnowledgeGraph


class TestKnowledgeGraph:
    def test_add_triple(self):
        kg = KnowledgeGraph()
        kg.add_triple("Einstein", "developed", "relativity")

        assert "relativity" in kg._adjacency["Einstein"]
        assert "Einstein" in kg._adjacency["relativity"]
        assert ("Einstein", "developed", "relativity") in kg._triples

    def test_get_neighbors_single_hop(self):
        kg = KnowledgeGraph()
        kg.add_triple("A", "relates_to", "B")
        kg.add_triple("B", "relates_to", "C")
        kg.add_triple("C", "relates_to", "D")

        neighbors = kg.get_neighbors("A", max_hops=1)
        assert neighbors == {"B"}

    def test_get_neighbors_multi_hop(self):
        kg = KnowledgeGraph()
        kg.add_triple("A", "relates_to", "B")
        kg.add_triple("B", "relates_to", "C")
        kg.add_triple("C", "relates_to", "D")

        neighbors = kg.get_neighbors("A", max_hops=2)
        assert "B" in neighbors
        assert "C" in neighbors
        assert "D" not in neighbors

    def test_document_links(self):
        kg = KnowledgeGraph()
        kg.add_document_link("Einstein", "doc_0")
        kg.add_document_link("Einstein", "doc_1")
        kg.add_document_link("Bohr", "doc_2")

        assert kg.get_related_documents("Einstein") == ["doc_0", "doc_1"]
        assert kg.get_related_documents("Bohr") == ["doc_2"]
        assert kg.get_related_documents("unknown") == []

    @pytest.mark.asyncio
    async def test_build_from_documents(self):
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "Einstein|developed|relativity\nNewton|discovered|gravity"

        kg = KnowledgeGraph()
        await kg.build_from_documents(["some science text"], mock_llm)

        assert ("Einstein", "developed", "relativity") in kg._triples
        assert ("Newton", "discovered", "gravity") in kg._triples
        assert "relativity" in kg._adjacency["Einstein"]
        assert "doc_0" in kg.get_related_documents("Einstein")
        assert "doc_0" in kg.get_related_documents("gravity")


class TestGraphRAGRetriever:
    @pytest.mark.asyncio
    async def test_retrieve_merges_graph_and_vector(self):
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "Einstein, relativity"

        mock_retriever = AsyncMock()
        mock_retriever.retrieve.return_value = ["chunk1", "chunk2"]

        kg = KnowledgeGraph()
        kg.add_triple("Einstein", "developed", "relativity")
        kg.add_document_link("Einstein", "doc_0")
        kg.add_document_link("relativity", "doc_1")

        graphrag = GraphRAGRetriever(
            retriever=mock_retriever, llm=mock_llm, knowledge_graph=kg
        )
        results = await graphrag.retrieve("Tell me about Einstein", top_k=5)

        # Should contain vector results and graph docs, deduplicated
        assert "chunk1" in results
        assert "chunk2" in results
        assert "doc_0" in results
        assert "doc_1" in results
        # No duplicates
        assert len(results) == len(set(results))

    @pytest.mark.asyncio
    async def test_retrieve_with_graph_returns_metadata(self):
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "Einstein"

        mock_retriever = AsyncMock()
        mock_retriever.retrieve.return_value = ["chunk1"]

        kg = KnowledgeGraph()
        kg.add_triple("Einstein", "developed", "relativity")
        kg.add_document_link("Einstein", "doc_0")

        graphrag = GraphRAGRetriever(
            retriever=mock_retriever, llm=mock_llm, knowledge_graph=kg
        )
        results, metadata = await graphrag.retrieve_with_graph(
            "Tell me about Einstein", top_k=5
        )

        assert isinstance(results, list)
        assert isinstance(metadata, dict)
        assert "entities_extracted" in metadata
        assert "graph_docs" in metadata
        assert "traversal_hops" in metadata
        assert "Einstein" in metadata["entities_extracted"]

    @pytest.mark.asyncio
    async def test_retrieve_without_knowledge_graph(self):
        mock_llm = AsyncMock()
        mock_retriever = AsyncMock()
        mock_retriever.retrieve.return_value = ["chunk1", "chunk2", "chunk3"]

        graphrag = GraphRAGRetriever(
            retriever=mock_retriever, llm=mock_llm, knowledge_graph=None
        )
        results = await graphrag.retrieve("some query", top_k=3)

        assert results == ["chunk1", "chunk2", "chunk3"]
        # LLM should not be called when there is no knowledge graph
        mock_llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_custom_max_hops(self):
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "A"

        mock_retriever = AsyncMock()
        mock_retriever.retrieve.return_value = []

        kg = KnowledgeGraph()
        kg.add_triple("A", "to", "B")
        kg.add_triple("B", "to", "C")
        kg.add_triple("C", "to", "D")
        kg.add_document_link("D", "doc_deep")
        kg.add_document_link("B", "doc_shallow")

        # max_hops=1 should only reach B, not C or D
        graphrag = GraphRAGRetriever(
            retriever=mock_retriever, llm=mock_llm, knowledge_graph=kg, max_hops=1
        )
        results = await graphrag.retrieve("query about A", top_k=10)

        assert "doc_shallow" in results
        assert "doc_deep" not in results

        # max_hops=3 should reach D
        graphrag_deep = GraphRAGRetriever(
            retriever=mock_retriever, llm=mock_llm, knowledge_graph=kg, max_hops=3
        )
        results_deep = await graphrag_deep.retrieve("query about A", top_k=10)

        assert "doc_deep" in results_deep
        assert "doc_shallow" in results_deep
