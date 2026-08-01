"""OKF → WorldModel graph adapter (#825).

Open Knowledge Format bundles cross-link via ordinary relative Markdown links,
which makes a bundle an *explicit* graph on disk. This adapter maps the
``Document`` list produced by :class:`~synapsekit.loaders.okf.OpenKnowledgeFormatLoader`
(which already resolved cross-links into ``metadata["linked_concepts"]``)
directly onto SynapseKit's :mod:`~synapsekit.retrieval.world_model` graph — one
node per concept, one edge per resolved link — with **no** LLM/heuristic
extraction.

Node identity is derived deterministically from ``concept_path`` and written
straight into the backend, *bypassing* the entity resolver: OKF ids are already
explicit and canonical, so the fuzzy/token-overlap merging a resolver does (e.g.
the mesh's cross-project resolver would fold ``tables/orders.md`` into
``tables/customers.md``) must not apply. The build is idempotent — re-ingesting
the same bundle merges onto the same ids instead of duplicating.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ..loaders.base import Document
from .world_model import (
    EntityResolver,
    GraphBackend,
    InMemoryWorldGraphBackend,
    WorldModelEdge,
    WorldModelNode,
    _slug,
)

__all__ = ["okf_to_world_model"]

# Predicate used for a plain OKF cross-link.
_LINK_PREDICATE = "links_to"

_normalize = EntityResolver._normalize


def okf_to_world_model(
    documents: list[Document],
    backend: GraphBackend | None = None,
    *,
    link_predicate: str = _LINK_PREDICATE,
) -> GraphBackend:
    """Build a WorldModel graph from OKF loader ``Document`` objects.

    Args:
        documents: The output of ``OpenKnowledgeFormatLoader(...).load()`` —
            each carries ``concept_path`` and (when link resolution was on)
            ``linked_concepts`` in its metadata.
        backend: A graph backend to populate. Defaults to a fresh
            :class:`InMemoryWorldGraphBackend`. Any backend exposing the
            in-memory storage model (in-memory, Kuzu, Neo4j) is supported.
        link_predicate: Predicate for cross-link edges.

    Returns:
        The populated ``backend`` (the same instance passed in, or the new one).

    Notes:
        Node ids are ``_slug(concept_path)`` — deterministic and path-unique, so
        the same bundle always yields the same graph and re-ingestion is
        idempotent. Frontmatter (``resource``/``tags``/``timestamp``/``okf_type``)
        is mapped onto node ``metadata``; ``timestamp`` also seeds edge
        ``valid_at``. Links to concepts not present in ``documents`` (e.g. a
        skipped ``index.md``) resolve to no edge.
    """
    backend = backend or InMemoryWorldGraphBackend()
    if not _supports_direct_insert(backend):
        raise TypeError(
            f"{type(backend).__name__} does not expose the world-model storage "
            "model required by okf_to_world_model."
        )

    # Pass 1 — one node per concept. Nodes must exist before edges so both
    # endpoints of every cross-link resolve.
    node_ids: dict[str, str] = {}
    for doc in documents:
        concept_path = doc.metadata.get("concept_path")
        if not concept_path:
            continue
        node_ids[concept_path] = _upsert_concept_node(backend, doc, concept_path)

    # Pass 2 — one edge per resolved in-bundle cross-link.
    for doc in documents:
        concept_path = doc.metadata.get("concept_path")
        if not concept_path:
            continue
        subject_id = node_ids[concept_path]
        valid_at = _coerce_datetime(doc.metadata.get("timestamp"))
        for target in doc.metadata.get("linked_concepts", []):
            object_id = node_ids.get(target)
            if object_id is None:
                continue  # link to a concept that wasn't ingested (e.g. index.md)
            _upsert_link_edge(
                backend, subject_id, object_id, link_predicate, concept_path, valid_at
            )

    return backend


def _supports_direct_insert(backend: GraphBackend) -> bool:
    return all(
        hasattr(backend, attr)
        for attr in ("nodes", "edges", "_name_to_id", "_documents_by_node", "_adjacency")
    )


def _upsert_concept_node(backend: Any, doc: Document, concept_path: str) -> str:
    node_id = _slug(concept_path)
    okf_type = str(doc.metadata.get("okf_type") or "concept")
    title = str(doc.metadata.get("title") or concept_path)

    aliases = {title}
    aliases.update(str(tag) for tag in doc.metadata.get("tags") or [])

    metadata: dict[str, Any] = {"concept_path": concept_path, "okf_type": okf_type}
    for key in ("resource", "tags", "title", "description", "timestamp"):
        value = doc.metadata.get(key)
        if value is not None:
            metadata[key] = value

    now = datetime.now(UTC)
    node = backend.nodes.get(node_id)
    if node is None:
        node = WorldModelNode(
            id=node_id,
            name=title,
            type=okf_type,
            aliases=set(aliases),
            provenance={concept_path},
            created_at=now,
            updated_at=now,
            metadata=metadata,
        )
        backend.nodes[node_id] = node
    else:
        node.name = title
        node.type = okf_type
        node.aliases.update(aliases)
        node.provenance.add(concept_path)
        node.metadata.update(metadata)
        node.updated_at = now

    # Index concept_path + title + tags so a query can seed by any of them,
    # without going through the (possibly merging) resolver.
    backend._name_to_id[_normalize(concept_path)] = node_id
    for alias in aliases:
        backend._name_to_id[_normalize(alias)] = node_id
    backend._documents_by_node[node_id].add(concept_path)

    _persist(backend, "_persist_entity", node)
    _persist(backend, "_persist_document", concept_path)
    return node_id


def _upsert_link_edge(
    backend: Any,
    subject_id: str,
    object_id: str,
    predicate: str,
    doc_id: str,
    valid_at: datetime | None,
) -> None:
    edge_id = f"{subject_id}:{_slug(predicate)}:{object_id}"
    now = datetime.now(UTC)
    edge = backend.edges.get(edge_id)
    if edge is None:
        edge = WorldModelEdge(
            id=edge_id,
            subject_id=subject_id,
            predicate=predicate,
            object_id=object_id,
            valid_at=valid_at,
            provenance={doc_id},
            created_at=now,
            updated_at=now,
            metadata={"okf_link": True},
        )
        backend.edges[edge_id] = edge
    else:
        edge.provenance.add(doc_id)
        edge.updated_at = now
        if edge.valid_at is None:
            edge.valid_at = valid_at

    backend._adjacency[subject_id].add(object_id)
    backend._adjacency[object_id].add(subject_id)
    backend._documents_by_node[subject_id].add(doc_id)
    backend._documents_by_node[object_id].add(doc_id)

    _persist(backend, "_persist_edge", edge)


def _persist(backend: Any, method: str, arg: Any) -> None:
    """Call a backend's persistence hook (e.g. Kuzu) when it has one."""
    hook = getattr(backend, method, None)
    if callable(hook):
        hook(arg)


def _coerce_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None
