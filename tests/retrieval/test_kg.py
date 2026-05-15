import pytest
from unittest.mock import AsyncMock

from synapsekit.retrieval.kg.backends import NetworkXStore
from synapsekit.retrieval.kg.builder import KnowledgeGraphBuilder
from synapsekit.retrieval.kg.retriever import KGRetriever


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
