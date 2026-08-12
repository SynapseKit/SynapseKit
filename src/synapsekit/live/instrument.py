"""Auto-instrument SynapseKit subsystems so their activity streams to Live.

Mirrors the observe LLM auto-instrumentation: wrap key async methods so each
call publishes a bus event with timing + a few attributes. It covers the paths
that observe spans miss — tools (and MCP tools), agent memory / DB, knowledge
graphs, and the knowledge mesh — by patching base classes (every subclass at
once, plus future ones via ``__init_subclass__``) or concrete classes.

Properties:
  * **Async-preserving** — async methods stay coroutines (asserted in tests).
  * **Near-zero when off** — the wrapper early-returns to the original when the
    bus is disabled; no timing, no publish.
  * **Idempotent** — a sentinel prevents double-wrapping.
  * **No new dependencies** — stdlib only.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import time
import traceback
from collections.abc import Callable
from typing import Any

from .bus import bus

_SENTINEL = "__synapsekit_live_wrapped__"
_HOOK_FLAG = "__synapsekit_live_hooked__"
_instrumented = False

AttrsFn = Callable[[Any, tuple[Any, ...], dict[str, Any]], dict[str, Any]]


def _publish(kind: str, start: float, status: str, attrs: dict[str, Any], tb: str | None) -> None:
    if tb:
        attrs = {**attrs, "traceback": tb[-1800:]}
    bus.publish(
        {
            "kind": kind,
            "name": kind,
            "status": status,
            "duration_ms": round((time.perf_counter() - start) * 1000, 3),
            "attributes": attrs,
        }
    )


def _make_wrapper(orig: Any, kind: str, attrs_fn: AttrsFn) -> Any:
    @functools.wraps(orig)
    async def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not bus.enabled:
            return await orig(self, *args, **kwargs)
        start, status, tb = time.perf_counter(), "ok", None
        try:
            return await orig(self, *args, **kwargs)
        except Exception:
            status, tb = "error", traceback.format_exc()
            raise
        finally:
            attrs: dict[str, Any] = {}
            with contextlib.suppress(Exception):
                attrs = attrs_fn(self, args, kwargs) or {}
            _publish(kind, start, status, attrs, tb)

    setattr(wrapped, _SENTINEL, True)
    return wrapped


def _make_sync_wrapper(orig: Any, kind: str, attrs_fn: AttrsFn) -> Any:
    @functools.wraps(orig)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not bus.enabled:
            return orig(self, *args, **kwargs)
        start, status, tb = time.perf_counter(), "ok", None
        try:
            return orig(self, *args, **kwargs)
        except Exception:
            status, tb = "error", traceback.format_exc()
            raise
        finally:
            attrs: dict[str, Any] = {}
            with contextlib.suppress(Exception):
                attrs = attrs_fn(self, args, kwargs) or {}
            _publish(kind, start, status, attrs, tb)

    setattr(wrapped, _SENTINEL, True)
    return wrapped


def _patch(cls: type, method_name: str, kind: str, attrs_fn: AttrsFn) -> None:
    """Wrap a method defined *on this class* (not inherited). Async or sync."""
    method = cls.__dict__.get(method_name)
    if method is None or getattr(method, _SENTINEL, False):
        return
    if inspect.iscoroutinefunction(method):
        setattr(cls, method_name, _make_wrapper(method, kind, attrs_fn))
    elif callable(method):
        setattr(cls, method_name, _make_sync_wrapper(method, kind, attrs_fn))


def _all_subclasses(base: type) -> list[type]:
    seen: set[type] = set()
    stack = list(base.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())
    return list(seen)


def _instrument_hierarchy(base: type, method_name: str, kind: str, attrs_fn: AttrsFn) -> None:
    """Patch ``base`` + every existing subclass, and hook future subclasses."""
    _patch(base, method_name, kind, attrs_fn)
    for sub in _all_subclasses(base):
        _patch(sub, method_name, kind, attrs_fn)

    hook_key = f"{_HOOK_FLAG}_{method_name}"
    if getattr(base, hook_key, False):
        return
    original_isc = base.__dict__.get("__init_subclass__")

    def __init_subclass__(cls: type, **kwargs: Any) -> None:  # noqa: N807
        if original_isc is not None:
            original_isc.__get__(None, cls)(**kwargs)
        _patch(cls, method_name, kind, attrs_fn)

    base.__init_subclass__ = classmethod(__init_subclass__)  # type: ignore[assignment]
    setattr(base, hook_key, True)


def _const(**fields: Any) -> AttrsFn:
    """An attrs function that always returns the same constant fields."""
    return lambda self, args, kwargs: dict(fields)


def _try(fn: Callable[[], None]) -> None:
    """Run one instrumentation step; never let a missing module break the rest."""
    with contextlib.suppress(Exception):
        fn()


def _snapshot_graph(backend: Any, limit: int = 60) -> None:
    """Read an in-memory graph backend and push its nodes/edges to the canvas."""
    if backend is None:
        return
    nodes_src = getattr(backend, "nodes", None)
    if nodes_src is None:
        nodes_src = getattr(backend, "_nodes", None)
    if not nodes_src:
        return  # remote backend (Neo4j etc.) — nothing in memory to read
    edges_src = getattr(backend, "edges", None)
    if edges_src is None:
        edges_src = getattr(backend, "_edges", None)

    node_items = list(nodes_src.values())[:limit]
    node_ids = {getattr(n, "id", None) for n in node_items}
    nodes = [
        {
            "id": n.id,
            "label": getattr(n, "name", None) or getattr(n, "label", n.id),
            "group": "graph",
        }
        for n in node_items
    ]
    edges = []
    for e in edges_src.values() if edges_src else []:
        src = getattr(e, "subject_id", None) or getattr(e, "source", None)
        dst = getattr(e, "object_id", None) or getattr(e, "target", None)
        if src in node_ids and dst in node_ids:
            edges.append([src, dst])

    from . import publish_graph

    publish_graph(nodes, edges)


def _wrap_ingest(cls: type, method_name: str, get_backend: Callable[[Any], Any]) -> None:
    """Wrap an ingest method to publish a graph.ingest event + a graph snapshot."""
    orig = cls.__dict__.get(method_name)
    if orig is None or getattr(orig, _SENTINEL, False) or not inspect.iscoroutinefunction(orig):
        return

    @functools.wraps(orig)
    async def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not bus.enabled:
            return await orig(self, *args, **kwargs)
        start, status, tb = time.perf_counter(), "ok", None
        try:
            return await orig(self, *args, **kwargs)
        except Exception:
            status, tb = "error", traceback.format_exc()
            raise
        finally:
            _publish("graph.ingest", start, status, {"graph": type(self).__name__}, tb)
            with contextlib.suppress(Exception):
                _snapshot_graph(get_backend(self))

    setattr(wrapped, _SENTINEL, True)
    setattr(cls, method_name, wrapped)


def instrument_all() -> None:
    """Instrument every supported subsystem. Idempotent; safe to call repeatedly."""
    global _instrumented
    if _instrumented:
        return
    _instrumented = True

    # -- Tools (covers all 47+ tools AND MCP tools, which subclass BaseTool) --
    def tools() -> None:
        from ..agents.base import BaseTool

        _instrument_hierarchy(
            BaseTool,
            "run",
            "tool.call",
            lambda self, a, k: {
                "tool": getattr(self, "name", type(self).__name__),
                **({"operation": k["operation"]} if k.get("operation") else {}),
            },
        )

    # -- Agent memory / DB (the logical read/write, whatever the backend) --
    def memory() -> None:
        from ..memory.agent_memory import AgentMemory

        _patch(AgentMemory, "store", "memory.write", lambda self, a, k: {"op": "store"})
        _patch(AgentMemory, "recall", "memory.read", lambda self, a, k: {"op": "recall"})

    # -- Knowledge graph: WorldModelRAG (user-facing) --
    def world_model() -> None:
        from ..retrieval.world_model import WorldModelRAG

        _patch(WorldModelRAG, "query", "graph.query", lambda self, a, k: {"graph": "world_model"})
        _patch(
            WorldModelRAG, "retrieve", "graph.query", lambda self, a, k: {"graph": "world_model"}
        )
        # ingest also snapshots the entity graph onto the Live canvas
        _wrap_ingest(WorldModelRAG, "ingest", lambda s: getattr(s, "graph_backend", None))

    # -- Knowledge graph: property-graph backends --
    def property_graph() -> None:
        from ..retrieval import property_graph as pg

        for name in ("NetworkXPropertyGraphBackend", "Neo4jPropertyGraphBackend"):
            cls = getattr(pg, name, None)
            if cls is not None:
                _patch(cls, "search", "graph.query", _const(backend=name))
                # the backend *is* self here, so snapshot its own nodes/edges
                _wrap_ingest(cls, "ingest", lambda s: s)

    # -- Knowledge mesh --
    def mesh() -> None:
        from ..mesh.core import KnowledgeMesh

        _patch(KnowledgeMesh, "query", "mesh.query", lambda self, a, k: {})
        _patch(KnowledgeMesh, "ingest_okf", "mesh.ingest", lambda self, a, k: {"format": "okf"})

    # -- Data loaders (no shared base class → patch the common concrete loaders) --
    def loaders() -> None:
        from .. import loaders as loaders_pkg

        for name in (
            "TextLoader",
            "CSVLoader",
            "JSONLoader",
            "MarkdownLoader",
            "DirectoryLoader",
            "PDFLoader",
            "HTMLLoader",
            "YAMLLoader",
        ):
            cls = getattr(loaders_pkg, name, None)
            if cls is None:
                continue
            for method in ("load", "aload"):
                _patch(cls, method, "loader.load", _const(loader=name))

    # -- Embeddings --
    def embeddings() -> None:
        from ..embeddings.backend import SynapsekitEmbeddings

        _patch(SynapsekitEmbeddings, "embed", "embeddings.embed", lambda self, a, k: {})
        with contextlib.suppress(Exception):
            from ..embeddings.onnx import ONNXEmbeddings

            _patch(ONNXEmbeddings, "embed", "embeddings.embed", _const(backend="onnx"))

    # -- Budget guard → live budget gauge --
    def budget() -> None:
        from ..observability.budget_guard import BudgetGuard

        _patch(
            BudgetGuard,
            "record_spend",
            "budget",
            lambda self, a, k: {
                "spent": round(getattr(self, "_daily_spend", 0.0), 6),
                "limit": getattr(getattr(self, "_limits", None), "daily", None),
                "cost": (a[0] if a else k.get("cost")),
            },
        )

    # -- Signed audit log --
    def audit() -> None:
        from ..observability.audit_log import AuditLog

        _patch(
            AuditLog,
            "record",
            "audit",
            lambda self, a, k: {
                "model": (a[0] if a else k.get("model")),
                "signed": True,
            },
        )

    # -- Agent swarm auctions --
    def swarm() -> None:
        from ..observability.metrics import PrometheusMetrics

        _patch(PrometheusMetrics, "record_swarm_win", "swarm", _const(event="win"))

    # -- Self-evolving agent: each evolution cycle and rollback --
    def agent_evolution() -> None:
        from ..agents.self_improving import SelfImprovingAgent

        def evolve_attrs(self: Any, a: tuple[Any, ...], k: dict[str, Any]) -> dict[str, Any]:
            # attrs are read after the cycle, so the newest audit entry is the
            # patch this cycle produced (accepted canary, or the last blocked decoy).
            attrs: dict[str, Any] = {"agent_id": getattr(self, "agent_id", "?")}
            hist = self.evolution_history(limit=1)
            if hist:
                p = hist[0]
                attrs.update(
                    {
                        "patch_id": p.patch_id[:8],
                        "patch_status": p.status,
                        "directive": p.metadata.get("directive"),
                        "eval_score": p.eval_score,
                        "baseline_score": p.baseline_score,
                        "block_reason": p.metadata.get("block_reason"),
                    }
                )
            return {kk: vv for kk, vv in attrs.items() if vv is not None}

        _patch(SelfImprovingAgent, "evolve", "agent.evolve", evolve_attrs)
        _patch(
            SelfImprovingAgent,
            "rollback",
            "agent.rollback",
            lambda self, a, k: {
                "agent_id": getattr(self, "agent_id", "?"),
                "rolled_back": (a[0][:8] if a and isinstance(a[0], str) else k.get("patch_id", "")),
                "reason": k.get("reason", "manual"),
            },
        )

    # -- Hive Mode: pooled-memory contribute / withdraw / suggestions --
    def hive() -> None:
        from ..hive.client import HiveClient

        scope = lambda self, a, k: {"scope_id": getattr(self, "scope_id", "?")}  # noqa: E731
        _patch(HiveClient, "contribute", "hive.contribute", scope)
        _patch(HiveClient, "withdraw", "hive.withdraw", scope)
        _patch(HiveClient, "suggestions_for", "hive.suggestions", scope)

    for step in (
        tools,
        memory,
        world_model,
        property_graph,
        mesh,
        loaders,
        embeddings,
        budget,
        audit,
        swarm,
        agent_evolution,
        hive,
    ):
        _try(step)
