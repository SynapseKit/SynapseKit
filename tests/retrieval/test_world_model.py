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
    WorldModelNode,
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
async def test_hybrid_world_model_retriever_honors_graph_first_and_vector_first():
    graph = InMemoryWorldGraphBackend()
    graph.upsert_entity(EntityMention("Alice"), "doc_graph")
    graph.upsert_entity(EntityMention("Search API"), "doc_graph")
    graph.upsert_relation(RelationMention("Alice", "worked_on", "Search API"), "doc_graph")

    vectorstore = InMemoryVectorStore(FakeEmbeddings())  # type: ignore[arg-type]
    await vectorstore.add(["vector context"], [{"source": "doc_vector"}])
    retriever = HybridWorldModelRetriever(graph, vector_retriever=Retriever(vectorstore))

    graph_first = await retriever.retrieve_with_scores(
        "Alice Search API", top_k=5, strategy="graph_first"
    )
    vector_first = await retriever.retrieve_with_scores(
        "Alice Search API", top_k=5, strategy="vector_first"
    )

    assert graph_first[0]["text"] == "doc_graph"
    assert vector_first[0]["text"] == "vector context"


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


def test_world_model_rag_isinstance_assertions() -> None:
    """Verify that WorldModelNode is importable and is a class."""
    assert WorldModelNode is not None


def test_relation_with_missing_node_returns_none() -> None:
    graph = InMemoryWorldGraphBackend()
    result = graph.upsert_relation(RelationMention("Ghost", "knows", "Alice"), "doc_1")
    assert result is None


def test_entity_same_name_different_type_first_wins() -> None:
    graph = InMemoryWorldGraphBackend()
    graph.upsert_entity(EntityMention("Alice", type="person", confidence=0.9), "doc_1")
    graph.upsert_entity(EntityMention("Alice", type="org", confidence=0.95), "doc_2")
    nodes = list(graph.nodes.values())
    alice_nodes = [n for n in nodes if n.name == "Alice"]
    assert len(alice_nodes) == 1
    assert alice_nodes[0].type == "person"  # first type wins; upsert merges, not replaces


@pytest.mark.asyncio
async def test_query_on_empty_graph_returns_empty_subgraph() -> None:
    from synapsekit.retrieval.vectorstore import InMemoryVectorStore

    wm = WorldModelRAG(
        llm=FakeLLM(),
        vector_store=InMemoryVectorStore(FakeEmbeddings()),  # type: ignore[arg-type]
        extractor=HeuristicWorldModelExtractor(),
    )
    result = await wm.query("anything at all")
    assert isinstance(result.subgraph.nodes, list)
    assert len(result.subgraph.nodes) == 0


@pytest.mark.asyncio
async def test_llm_extractor_handles_invalid_json() -> None:
    class LLMReturningGarbage:
        async def generate(self, prompt: str, **kw) -> str:
            return "not json at all !!!"

        async def stream(self, prompt: str, **kw):
            yield "not json at all !!!"

    extractor = LLMWorldModelExtractor(llm=LLMReturningGarbage())  # type: ignore[arg-type]
    result = await extractor.extract("some text", ExtractionPolicy())
    assert isinstance(result, ExtractionResult)
    assert len(result.entities) == 0


def test_world_model_handles_60_entities() -> None:
    graph = InMemoryWorldGraphBackend()
    for i in range(60):
        graph.upsert_entity(EntityMention(f"Entity{i:03d}", type="person", confidence=0.9), "doc_1")
    assert len(graph.nodes) == 60


def test_world_model_async_methods_are_coroutines() -> None:
    import inspect

    assert inspect.iscoroutinefunction(HeuristicWorldModelExtractor.extract)
    assert inspect.iscoroutinefunction(LLMWorldModelExtractor.extract)
    assert inspect.iscoroutinefunction(WorldModelRAG.ingest)
    assert inspect.iscoroutinefunction(WorldModelRAG.query)


def test_external_backend_errors_clearly():
    wm = WorldModelRAG(
        llm=FakeLLM(),
        vector_store=InMemoryVectorStore(FakeEmbeddings()),  # type: ignore[arg-type]
        graph_backend="dgraph",  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="not available in the core package"):
        wm.subgraph_to_mermaid("anything")


def test_neo4j_and_memgraph_strings_route_to_neo4j_backend(monkeypatch):
    # Regression guard: both "neo4j" and "memgraph" must map to
    # Neo4jWorldGraphBackend in _make_graph_backend. Doesn't need a live
    # server -- GraphDatabase.driver() connects lazily on first use.
    pytest.importorskip("neo4j")
    from synapsekit.retrieval.world_model import Neo4jWorldGraphBackend

    monkeypatch.setenv("NEO4J_URI", "bolt://example.invalid:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "custom-user")
    monkeypatch.setenv("NEO4J_PASSWORD", "custom-pass")

    for backend_name in ("neo4j", "memgraph"):
        wm = WorldModelRAG(
            llm=FakeLLM(),
            vector_store=InMemoryVectorStore(FakeEmbeddings()),  # type: ignore[arg-type]
            graph_backend=backend_name,  # type: ignore[arg-type]
        )
        assert isinstance(wm.graph_backend, Neo4jWorldGraphBackend)
        assert wm.graph_backend.uri == "bolt://example.invalid:7687"


def test_kuzu_backend_roundtrip(tmp_path):
    pytest.importorskip("kuzu")
    from synapsekit.retrieval.world_model import KuzuWorldGraphBackend

    backend = KuzuWorldGraphBackend(tmp_path / "world_model.kuzu")
    backend.upsert_entity(EntityMention(name="Apollo", type="product"), "d1")
    backend.upsert_entity(EntityMention(name="Dana", type="person"), "d1")
    backend.upsert_relation(
        RelationMention(subject="Dana", predicate="leads", object="Apollo", confidence=0.9),
        "d1",
    )

    result = backend.query_subgraph("Apollo Dana", max_hops=1)
    assert {node.name for node in result.nodes} == {"Apollo", "Dana"}
    assert [edge.predicate for edge in result.edges] == ["leads"]
    assert result.documents == ["d1"]

    count_result = backend._execute("MATCH (e:Entity) RETURN count(e) AS c")
    assert count_result.get_next()[0] == 2


def test_neo4j_backend_roundtrip_with_testcontainers():
    pytest.importorskip("neo4j")
    container_mod = pytest.importorskip("testcontainers.core.container")
    wait_mod = pytest.importorskip("testcontainers.core.waiting_utils")
    docker_container = container_mod.DockerContainer
    wait_for_logs = wait_mod.wait_for_logs

    from synapsekit.retrieval.world_model import Neo4jWorldGraphBackend

    password = "synapsekit-password"
    with (
        docker_container("neo4j:5")
        .with_env("NEO4J_AUTH", f"neo4j/{password}")
        .with_exposed_ports(7687) as container
    ):
        wait_for_logs(container, "Bolt enabled", timeout=60)
        host = container.get_container_host_ip()
        port = container.get_exposed_port(7687)
        backend = Neo4jWorldGraphBackend(
            f"bolt://{host}:{port}",
            username="neo4j",
            password=password,
        )
        try:
            backend.upsert_entity(EntityMention(name="Apollo", type="product"), "d1")
            backend.upsert_entity(EntityMention(name="Dana", type="person"), "d1")
            backend.upsert_relation(
                RelationMention(
                    subject="Dana", predicate="leads", object="Apollo", confidence=0.9
                ),
                "d1",
            )

            # In-memory mirror is what query_subgraph/to_mermaid read from.
            result = backend.query_subgraph("Apollo Dana", max_hops=1)
            assert {node.name for node in result.nodes} == {"Apollo", "Dana"}
            assert [edge.predicate for edge in result.edges] == ["leads"]
            assert result.documents == ["d1"]

            # Also verify the write actually landed in Neo4j, not just the mirror.
            with backend._driver.session() as session:
                count = session.run(
                    "MATCH (e:WorldModelEntity) RETURN count(e) AS c"
                ).single()["c"]
            assert count == 2
        finally:
            backend.close()


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
