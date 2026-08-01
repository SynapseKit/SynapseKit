"""Tests for the OKF → WorldModel graph adapter (#825) — real bundles, no mocks."""

import textwrap
from pathlib import Path

import pytest

from synapsekit import (
    HybridWorldModelRetriever,
    InMemoryWorldGraphBackend,
    OpenKnowledgeFormatLoader,
    okf_to_world_model,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8")


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    """Reuse the #824 bundle shape: orders → customers, a metric, an index stub."""
    root = tmp_path / "sales"
    _write(
        root / "tables" / "orders.md",
        """
        ---
        type: table
        title: Orders
        resource: bigquery://proj.sales.orders
        tags: [sales, fact]
        timestamp: "2026-01-02T00:00:00Z"
        ---
        # Orders

        Joins to [customers](/tables/customers.md).
        External [ref](https://example.com) and [gone](/tables/missing.md).
        """,
    )
    _write(
        root / "tables" / "customers.md",
        "---\ntype: table\ntitle: Customers\n---\n# Customers\n",
    )
    _write(
        root / "metrics" / "revenue.md",
        "---\ntype: metric\ntitle: Revenue\n---\n# Revenue\n\nSUM of [orders](/tables/orders.md).\n",
    )
    _write(root / "index.md", "---\ntype: index\n---\n# Nav\n\n[orders](/tables/orders.md)\n")
    return root


def _docs(bundle: Path):
    return OpenKnowledgeFormatLoader(bundle).load()


def _edge_pairs(backend: InMemoryWorldGraphBackend) -> set[tuple[str, str, str]]:
    id_to_path = {n.id: n.metadata["concept_path"] for n in backend.nodes.values()}
    return {
        (id_to_path[e.subject_id], e.predicate, id_to_path[e.object_id])
        for e in backend.edges.values()
    }


def test_graph_shape_nodes_edges_and_types(bundle: Path):
    backend = okf_to_world_model(_docs(bundle))

    by_path = {n.metadata["concept_path"]: n for n in backend.nodes.values()}
    assert set(by_path) == {"tables/orders.md", "tables/customers.md", "metrics/revenue.md"}
    assert by_path["tables/orders.md"].type == "table"
    assert by_path["metrics/revenue.md"].type == "metric"
    # Display name is the human title; concept_path stays in metadata.
    assert by_path["tables/orders.md"].name == "Orders"

    pairs = _edge_pairs(backend)
    assert ("tables/orders.md", "links_to", "tables/customers.md") in pairs
    assert ("metrics/revenue.md", "links_to", "tables/orders.md") in pairs
    # External links and links to a missing/absent concept produce no edge.
    assert len(pairs) == 2


def test_frontmatter_mapped_to_node_metadata(bundle: Path):
    backend = okf_to_world_model(_docs(bundle))
    orders = next(
        n for n in backend.nodes.values() if n.metadata["concept_path"] == "tables/orders.md"
    )
    assert orders.metadata["resource"] == "bigquery://proj.sales.orders"
    assert orders.metadata["tags"] == ["sales", "fact"]
    assert orders.metadata["okf_type"] == "table"
    # tags/title also become aliases so a query can seed by either.
    assert "sales" in orders.aliases
    assert "Orders" in orders.aliases
    # timestamp seeds the outgoing edge's valid_at.
    edge = next(iter(e for e in backend.edges.values() if e.subject_id == orders.id))
    assert edge.valid_at is not None


def test_deterministic_ids(bundle: Path):
    a = okf_to_world_model(_docs(bundle))
    b = okf_to_world_model(_docs(bundle))
    assert sorted(a.nodes) == sorted(b.nodes)
    assert sorted(a.edges) == sorted(b.edges)


def test_idempotent_reingestion(bundle: Path):
    backend = okf_to_world_model(_docs(bundle))
    n_nodes, n_edges = len(backend.nodes), len(backend.edges)
    # Re-ingest into the same backend → no duplicates.
    okf_to_world_model(_docs(bundle), backend)
    assert len(backend.nodes) == n_nodes
    assert len(backend.edges) == n_edges


def test_links_to_skipped_concept_produce_no_edge(bundle: Path):
    # index.md links to orders but is skipped by the loader, so no index node/edge.
    backend = okf_to_world_model(_docs(bundle))
    assert all("index.md" not in n.metadata["concept_path"] for n in backend.nodes.values())


def test_resolve_links_disabled_yields_no_edges(bundle: Path):
    docs = OpenKnowledgeFormatLoader(bundle, resolve_links=False).load()
    backend = okf_to_world_model(docs)
    assert len(backend.nodes) == 3
    assert len(backend.edges) == 0


def test_defaults_to_in_memory_backend(bundle: Path):
    backend = okf_to_world_model(_docs(bundle))
    assert isinstance(backend, InMemoryWorldGraphBackend)


class _EmptyVectorRetriever:
    """Hand-written retriever returning no vector hits (graph-only assertions)."""

    async def retrieve_with_scores(self, query, top_k=5, metadata_filter=None):
        return []


async def test_graph_first_query_returns_linked_concept_documents(bundle: Path):
    backend = okf_to_world_model(_docs(bundle))
    retriever = HybridWorldModelRetriever(
        backend, _EmptyVectorRetriever(), strategy="graph_first", max_hops=2
    )
    docs = await retriever.retrieve("Orders")
    # Seeding at the orders concept expands to the linked customers concept;
    # both concept paths are returned as source documents.
    assert "tables/orders.md" in docs
    assert "tables/customers.md" in docs


async def test_knowledge_mesh_ingest_okf_end_to_end(bundle: Path, tmp_path: Path):
    from synapsekit.mesh import KnowledgeMesh, MeshConfig

    mesh = KnowledgeMesh(
        MeshConfig(
            roots=[bundle],
            state_dir=tmp_path / "mesh_state",
            vector_backend="memory",
            graph_backend="memory",
        )
    )
    count = await mesh.ingest_okf(bundle)
    assert count == 3

    # The explicit OKF graph landed on the mesh's world model...
    backend = mesh.rag.graph_backend
    paths = {n.metadata["concept_path"] for n in backend.nodes.values()}
    assert paths == {"tables/orders.md", "tables/customers.md", "metrics/revenue.md"}

    # ...and a graph_first query over the mesh's WorldModelRAG stack expands
    # from the seed concept to its linked concept.
    hits = await mesh.rag.retriever.retrieve("Orders", top_k=5)
    assert "tables/customers.md" in hits


def test_kuzu_backend_parity(bundle: Path):
    kuzu = pytest.importorskip("kuzu")
    assert kuzu is not None
    import tempfile

    from synapsekit import KuzuWorldGraphBackend

    with tempfile.TemporaryDirectory() as tmp:
        backend = KuzuWorldGraphBackend(Path(tmp) / "g.kuzu")
        okf_to_world_model(_docs(bundle), backend)
        pairs = _edge_pairs(backend)
        assert ("tables/orders.md", "links_to", "tables/customers.md") in pairs
        assert len(backend.nodes) == 3
