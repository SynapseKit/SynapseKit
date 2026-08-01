"""Tests for OpenKnowledgeFormatLoader (#824) — real bundles on tmp_path, no mocks."""

import inspect
import textwrap
from pathlib import Path

import pytest

from synapsekit import OKFLoader, OpenKnowledgeFormatLoader
from synapsekit.loaders import Document


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8")


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    """A small OKF bundle mirroring the spec example."""
    root = tmp_path / "sales"
    _write(
        root / "tables" / "orders.md",
        """
        ---
        type: table
        title: Orders
        description: One row per customer order.
        resource: bigquery://proj.sales.orders
        tags: [sales, fact]
        timestamp: "2026-01-02T00:00:00Z"
        ---
        # Orders

        Fact table. Joins to [customers](/tables/customers.md) on `customer_id`.
        See also [an external ref](https://example.com/spec) and [orphan](./missing.md).
        """,
    )
    _write(
        root / "tables" / "customers.md",
        """
        ---
        type: table
        title: Customers
        tags: [sales, dim]
        ---
        # Customers

        Dimension table.
        """,
    )
    _write(
        root / "metrics" / "revenue.md",
        """
        ---
        type: metric
        title: Revenue
        ---
        # Revenue

        SUM of [orders](/tables/orders.md) amount.
        """,
    )
    _write(
        root / "index.md",
        """
        ---
        type: index
        ---
        # Navigation
        """,
    )
    return root


def _by_path(docs: list[Document]) -> dict[str, Document]:
    return {d.metadata["concept_path"]: d for d in docs}


def test_loads_one_document_per_concept_with_metadata(bundle: Path):
    docs = OpenKnowledgeFormatLoader(bundle).load()
    by_path = _by_path(docs)

    # index.md skipped by default → 3 concepts.
    assert set(by_path) == {
        "tables/orders.md",
        "tables/customers.md",
        "metrics/revenue.md",
    }

    orders = by_path["tables/orders.md"]
    assert orders.metadata["source"] == "okf"
    assert orders.metadata["okf_type"] == "table"
    assert orders.metadata["title"] == "Orders"
    assert orders.metadata["resource"] == "bigquery://proj.sales.orders"
    assert orders.metadata["tags"] == ["sales", "fact"]
    assert orders.metadata["timestamp"] == "2026-01-02T00:00:00Z"
    assert orders.metadata["bundle_root"] == str(bundle.resolve())
    # Body has the frontmatter stripped.
    assert orders.text.startswith("# Orders")
    assert "type: table" not in orders.text
    # Raw frontmatter preserved for non-standard producer fields.
    assert orders.metadata["frontmatter"]["type"] == "table"


def test_cross_links_resolved_external_and_unresolved_excluded(bundle: Path):
    by_path = _by_path(OpenKnowledgeFormatLoader(bundle).load())

    # orders links to customers (in-bundle), an external URL, and a missing file.
    assert by_path["tables/orders.md"].metadata["linked_concepts"] == ["tables/customers.md"]
    # revenue links to orders via an absolute /tables/orders.md link.
    assert by_path["metrics/revenue.md"].metadata["linked_concepts"] == ["tables/orders.md"]
    # customers links to nothing.
    assert by_path["tables/customers.md"].metadata["linked_concepts"] == []


def test_resolve_links_can_be_disabled(bundle: Path):
    docs = OpenKnowledgeFormatLoader(bundle, resolve_links=False).load()
    assert all("linked_concepts" not in d.metadata for d in docs)
    # The link text still lives in the body.
    orders = _by_path(docs)["tables/orders.md"]
    assert "[customers](/tables/customers.md)" in orders.text


def test_index_stub_included_when_opted_in(bundle: Path):
    paths = {
        d.metadata["concept_path"]
        for d in OpenKnowledgeFormatLoader(bundle, include_index=True).load()
    }
    assert "index.md" in paths


def test_deterministic_ordering(bundle: Path):
    first = [d.metadata["concept_path"] for d in OpenKnowledgeFormatLoader(bundle).load()]
    second = [d.metadata["concept_path"] for d in OpenKnowledgeFormatLoader(bundle).load()]
    assert first == second == sorted(first)


def test_single_file_path(bundle: Path):
    docs = OpenKnowledgeFormatLoader(bundle / "tables" / "orders.md").load()
    assert len(docs) == 1
    assert docs[0].metadata["concept_path"] == "orders.md"
    # In single-file mode the bundle root is the file's own directory, so the
    # concept's absolute /tables/... cross-links can't resolve — best-effort.
    assert docs[0].metadata["linked_concepts"] == []


def test_single_file_relative_sibling_link_resolves(tmp_path: Path):
    _write(tmp_path / "customers.md", "---\ntype: table\n---\n# Customers\n")
    _write(
        tmp_path / "orders.md",
        "---\ntype: table\n---\n# Orders\n\nJoins [customers](customers.md).\n",
    )
    docs = OpenKnowledgeFormatLoader(tmp_path / "orders.md").load()
    # A relative sibling link resolves against the file's directory.
    assert docs[0].metadata["linked_concepts"] == ["customers.md"]


def test_non_recursive_only_top_level(bundle: Path):
    # Add a top-level concept; non-recursive should ignore the nested ones.
    _write(bundle / "overview.md", "---\ntype: doc\n---\n# Overview\n")
    paths = {
        d.metadata["concept_path"]
        for d in OpenKnowledgeFormatLoader(bundle, recursive=False).load()
    }
    assert paths == {"overview.md"}


def test_missing_type_skipped_by_default_emitted_when_not_required(tmp_path: Path, caplog):
    _write(tmp_path / "no_type.md", "---\ntitle: Untyped\n---\n# Untyped\n")

    with caplog.at_level("WARNING"):
        assert OpenKnowledgeFormatLoader(tmp_path).load() == []
    assert "no `type`" in caplog.text

    lenient = OpenKnowledgeFormatLoader(tmp_path, require_type=False).load()
    assert len(lenient) == 1
    assert lenient[0].metadata["okf_type"] is None


def test_empty_dir_and_non_md_ignored(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    assert OpenKnowledgeFormatLoader(tmp_path / "empty").load() == []

    _write(tmp_path / "notes.txt", "not markdown")
    _write(tmp_path / "data.json", "{}")
    assert OpenKnowledgeFormatLoader(tmp_path).load() == []


def test_malformed_frontmatter_warns_not_crashes(tmp_path: Path, caplog):
    _write(
        tmp_path / "broken.md",
        "---\ntype: table\n  bad: [unclosed\n---\n# Broken\n",
    )
    with caplog.at_level("WARNING"):
        docs = OpenKnowledgeFormatLoader(tmp_path, require_type=False).load()
    assert "malformed YAML" in caplog.text
    # Did not crash; body is still recoverable, type unknown.
    assert len(docs) == 1
    assert docs[0].metadata["okf_type"] is None
    assert docs[0].text.startswith("# Broken")


def test_no_frontmatter_body_preserved(tmp_path: Path):
    _write(tmp_path / "plain.md", "# Plain\n\nJust a body.\n")
    docs = OpenKnowledgeFormatLoader(tmp_path, require_type=False).load()
    assert len(docs) == 1
    assert docs[0].metadata["okf_type"] is None
    assert docs[0].text == "# Plain\n\nJust a body."


def test_aload_is_coroutine_and_matches_load(bundle: Path):
    assert inspect.iscoroutinefunction(OpenKnowledgeFormatLoader.aload)


async def test_aload_returns_same_documents(bundle: Path):
    loader = OpenKnowledgeFormatLoader(bundle)
    sync_docs = loader.load()
    async_docs = await loader.aload()
    assert [d.metadata["concept_path"] for d in async_docs] == [
        d.metadata["concept_path"] for d in sync_docs
    ]
    assert [d.text for d in async_docs] == [d.text for d in sync_docs]


def test_okf_alias_is_the_same_class():
    assert OKFLoader is OpenKnowledgeFormatLoader
