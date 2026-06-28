from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import numpy as np
import pytest

from synapsekit.llm.base import BaseLLM, LLMConfig
from synapsekit.retrieval.retriever import Retriever
from synapsekit.retrieval.vectorstore import InMemoryVectorStore
from synapsekit.retrieval.world_model import (
    EntityMention,
    EntityResolver,
    EventMention,
    ExtractionPolicy,
    ExtractionResult,
    HeuristicWorldModelExtractor,
    HybridWorldModelRetriever,
    InMemoryWorldGraphBackend,
    LLMWorldModelExtractor,
    RelationMention,
    WorldModelRAG,
)


class FakeEmbeddings:
    async def embed(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            rows.append(self._vector(text))
        return np.array(rows, dtype=np.float32)

    async def embed_one(self, text: str) -> np.ndarray:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        values = np.zeros(4, dtype=np.float32)
        for index, char in enumerate(text.encode("utf-8")):
            values[index % 4] += float(char)
        norm = np.linalg.norm(values)
        return values / norm if norm else values


class FakeLLM(BaseLLM):
    def __init__(self, response: str = "answer") -> None:
        super().__init__(
            LLMConfig(
                model="fake",
                api_key="",
                provider="fake",
            )
        )
        self.response = response
        self.prompts: list[str] = []

    async def stream(self, prompt: str, **kw) -> AsyncGenerator[str]:
        self.prompts.append(prompt)
        yield self.response


def test_in_memory_graph_resolves_aliases_and_tracks_provenance():
    graph = InMemoryWorldGraphBackend(EntityResolver())

    alice = graph.upsert_entity(
        EntityMention("Alice Chen", type="person", aliases=("@alice",)),
        "doc_1",
    )
    alias = graph.upsert_entity(EntityMention("@alice", type="person"), "doc_2")

    assert alias.id == alice.id
    assert graph.nodes[alice.id].provenance == {"doc_1", "doc_2"}


def test_temporal_query_filters_future_edges():
    graph = InMemoryWorldGraphBackend()
    graph.upsert_entity(EntityMention("Alice"), "doc_release")
    graph.upsert_entity(EntityMention("v1.5 release"), "doc_release")
    graph.upsert_relation(
        RelationMention(
            "Alice",
            "caused",
            "v1.5 release",
            valid_at=datetime(2026, 4, 1, tzinfo=UTC),
            causal=True,
        ),
        "doc_release",
    )

    before = graph.query_subgraph("Alice release", as_of="2026-03-01")
    after = graph.query_subgraph("Alice release", as_of="2026-05-01")

    assert before.edges == []
    assert len(after.edges) == 1
    assert after.edges[0].causal is True


def test_graph_mermaid_marks_causal_edges():
    graph = InMemoryWorldGraphBackend()
    graph.upsert_entity(EntityMention("Alice"), "doc_1")
    graph.upsert_entity(EntityMention("v1.5 release"), "doc_1")
    graph.upsert_relation(RelationMention("Alice", "caused", "v1.5 release", causal=True), "doc_1")

    mermaid = graph.to_mermaid(graph.query_subgraph("Alice"))

    assert "graph LR" in mermaid
    assert "caused (causal)" in mermaid
    assert "Alice" in mermaid


@pytest.mark.asyncio
async def test_llm_extractor_parses_strict_json():
    llm = FakeLLM(
        """
        {
          "entities": [{"name": "Alice", "type": "person", "aliases": ["@alice"], "confidence": 0.9}],
          "relations": [{"subject": "Alice", "predicate": "worked_on", "object": "API", "confidence": 0.8, "valid_at": "2026-03-01", "valid_until": null, "causal": false}],
          "events": [{"name": "launch", "participants": ["Alice"], "timestamp": "2026-03-15", "confidence": 0.7}]
        }
        """
    )
    extractor = LLMWorldModelExtractor(llm)

    result = await extractor.extract("Alice worked on API", ExtractionPolicy())

    assert result.entities[0].name == "Alice"
    assert result.entities[0].aliases == ("@alice",)
    assert result.relations[0].valid_at == datetime(2026, 3, 1, tzinfo=UTC)
    assert result.events[0].participants == ("Alice",)


@pytest.mark.asyncio
async def test_heuristic_extractor_finds_causal_temporal_relation():
    extractor = HeuristicWorldModelExtractor()

    result = await extractor.extract(
        "Alice worked on Search API in 2026. Search API led to v1.5 Release.",
        ExtractionPolicy(),
    )

    assert any(entity.name == "Alice" for entity in result.entities)
    assert any(relation.causal for relation in result.relations)
    assert result.events


@pytest.mark.asyncio
async def test_hybrid_world_model_retriever_fuses_graph_and_vector_results():
    graph = InMemoryWorldGraphBackend()
    graph.upsert_entity(EntityMention("Alice"), "doc_graph")
    graph.upsert_entity(EntityMention("Search API"), "doc_graph")
    graph.upsert_relation(RelationMention("Alice", "worked_on", "Search API"), "doc_graph")

    vectorstore = InMemoryVectorStore(FakeEmbeddings())  # type: ignore[arg-type]
    await vectorstore.add(["vector context"], [{"source": "doc_vector"}])
    vector = HybridWorldModelRetriever(graph, vector_retriever=Retriever(vectorstore))

    results = await vector.retrieve_with_scores("Alice Search API", top_k=5)
    texts = [result["text"] for result in results]

    assert "doc_graph" in texts
    assert "vector context" in texts


@pytest.mark.asyncio
async def test_world_model_rag_ingest_query_and_mermaid():
    llm = FakeLLM("Alice worked on Search API, which led to v1.5.")
    vectorstore = InMemoryVectorStore(FakeEmbeddings())  # type: ignore[arg-type]
    wm = WorldModelRAG(
        llm=llm,
        vector_store=vectorstore,
        extractor=HeuristicWorldModelExtractor(),
        retrieval_top_k=5,
    )

    await wm.ingest(
        [
            {
                "text": "Alice worked on Search API in 2026. Search API led to v1.5 Release.",
                "metadata": {"source": "release_notes"},
            }
        ]
    )
    result = await wm.query("What did Alice work on that led to v1.5?", as_of="2026-12-01")

    assert result.answer == "Alice worked on Search API, which led to v1.5."
    assert result.subgraph.nodes
    assert result.embeddings
    assert "World-model subgraph" in llm.prompts[-1]
    assert "release_notes" in {item["metadata"]["source"] for item in result.embeddings}
    assert "caused (causal)" in wm.subgraph_to_mermaid("Alice")


def test_external_backend_errors_clearly():
    wm = WorldModelRAG(
        llm=FakeLLM(),
        vector_store=InMemoryVectorStore(FakeEmbeddings()),  # type: ignore[arg-type]
        graph_backend="neo4j",
    )

    with pytest.raises(RuntimeError, match="not available in the core package"):
        wm.subgraph_to_mermaid("anything")


def test_event_ingestion_links_participants():
    graph = InMemoryWorldGraphBackend()
    graph.upsert_entity(EntityMention("Alice"), "doc_event")
    graph.add_event(
        EventMention(
            "v1.5 release",
            participants=("Alice",),
            timestamp=datetime(2026, 5, 1, tzinfo=UTC),
        ),
        "doc_event",
    )

    result = graph.query_subgraph("Alice release", as_of="2026-06-01")

    assert any(edge.predicate == "participated_in" for edge in result.edges)


@pytest.mark.asyncio
async def test_custom_extractor_is_used_during_ingest():
    class Extractor:
        async def extract(self, text: str, policy: ExtractionPolicy) -> ExtractionResult:
            return ExtractionResult(
                entities=[EntityMention("Deprecated API"), EntityMention("Billing Service")],
                relations=[RelationMention("Billing Service", "depends_on", "Deprecated API", 0.9)],
            )

    wm = WorldModelRAG(
        llm=FakeLLM(),
        vector_store=InMemoryVectorStore(FakeEmbeddings()),  # type: ignore[arg-type]
        extractor=Extractor(),
    )

    await wm.ingest(["Billing Service depends on Deprecated API"])
    subgraph = wm.graph_backend.query_subgraph("Which entities depend on Deprecated API?")

    assert any(edge.predicate == "depends_on" for edge in subgraph.edges)
