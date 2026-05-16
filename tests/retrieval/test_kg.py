from unittest.mock import AsyncMock

import pytest

pytest.importorskip("networkx")

from synapsekit.retrieval.kg.backends import NetworkXStore
from synapsekit.retrieval.kg.builder import KnowledgeGraphBuilder
from synapsekit.retrieval.kg.retriever import HybridKGRetriever, KGRetriever


@pytest.mark.asyncio
async def test_kg_builder_and_retrieval():
    # Mock LLM that returns fixed JSON for our test
    mock_llm = AsyncMock()
    # First call: extract entities for build
    # Second call: extract triples for build
    # Third call: extract entities for query

    mock_llm.generate.side_effect = [
        # Document 1 triples
        '[{"subject": "Apex Biotech", "predicate": "acquired", "object": "MedCorp", "confidence": 0.9}, {"subject": "MedCorp", "predicate": "developed", "object": "CardioDrug", "confidence": 0.8}]',
        # Query entities
        '["Apex Biotech"]',
    ]

    store = NetworkXStore()
    builder = KnowledgeGraphBuilder(llm=mock_llm, store=store)

    docs = ["Apex Biotech recently acquired MedCorp, the company that developed CardioDrug."]
    await builder.build_from_documents(docs, doc_ids=["doc_1"])

    # Check store directly
    neighbors = store.get_neighbors("Apex Biotech", max_hops=1)
    assert "MedCorp" in neighbors

    # Check graph traversal retrieval
    retriever = KGRetriever(store=store, builder=builder, max_hops=2)
    doc_ids = await retriever.retrieve("What drugs are associated with Apex Biotech?")

    # "Apex Biotech" -> "MedCorp" (doc_1)
    # "MedCorp" -> "CardioDrug" (doc_1)
    assert "doc_1" in doc_ids


@pytest.mark.asyncio
async def test_entity_fallback_handles_commas():
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = '"Acme, Inc.", Beta Corp'

    store = NetworkXStore()
    builder = KnowledgeGraphBuilder(llm=mock_llm, store=store)

    entities = await builder.extract_entities("Acme, Inc. works with Beta Corp")

    assert entities == ["Acme, Inc.", "Beta Corp"]


@pytest.mark.asyncio
async def test_extract_triples_invalid_json_returns_empty():
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = "not json"

    store = NetworkXStore()
    builder = KnowledgeGraphBuilder(llm=mock_llm, store=store)

    triples = await builder.extract_triples("A relates to B")

    assert triples == []


@pytest.mark.asyncio
async def test_build_skips_invalid_triples():
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = (
        '["oops", {"subject": "A", "predicate": "", "object": "B"}, '
        '{"subject": "A", "predicate": "relates", "object": "B", "confidence": 0.7}]'
    )

    store = NetworkXStore()
    builder = KnowledgeGraphBuilder(llm=mock_llm, store=store)

    await builder.build_from_documents(["doc"], doc_ids=["doc_1"])

    assert "B" in store.get_neighbors("A", max_hops=1)


def test_graph_cycle_does_not_loop_forever():
    store = NetworkXStore()
    store.add_triple("A", "relates", "B", 0.9)
    store.add_triple("B", "relates", "A", 0.9)

    neighbors = store.get_neighbors("A", max_hops=1)

    assert neighbors == {"B"}


@pytest.mark.asyncio
async def test_hybrid_retriever_merges_and_dedups():
    vector = AsyncMock()
    vector.retrieve = AsyncMock(return_value=["doc_1", "doc_2"])

    kg = AsyncMock()
    kg.retrieve = AsyncMock(return_value=["doc_2", "doc_3"])

    retriever = HybridKGRetriever(vector_retriever=vector, kg_retriever=kg)

    results = await retriever.retrieve("query", top_k=5)

    assert results == ["doc_1", "doc_2", "doc_3"]
