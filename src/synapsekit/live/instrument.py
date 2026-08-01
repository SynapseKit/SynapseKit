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
from collections.abc import Callable
from typing import Any

from .bus import bus

_SENTINEL = "__synapsekit_live_wrapped__"
_HOOK_FLAG = "__synapsekit_live_hooked__"
_instrumented = False

AttrsFn = Callable[[Any, tuple[Any, ...], dict[str, Any]], dict[str, Any]]


def _make_wrapper(orig: Any, kind: str, attrs_fn: AttrsFn) -> Any:
    @functools.wraps(orig)
    async def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not bus.enabled:
            return await orig(self, *args, **kwargs)
        start = time.perf_counter()
        status = "ok"
        try:
            return await orig(self, *args, **kwargs)
        except Exception:
            status = "error"
            raise
        finally:
            attrs: dict[str, Any] = {}
            with contextlib.suppress(Exception):
                attrs = attrs_fn(self, args, kwargs) or {}
            bus.publish(
                {
                    "kind": kind,
                    "name": kind,
                    "status": status,
                    "duration_ms": round((time.perf_counter() - start) * 1000, 3),
                    "attributes": attrs,
                }
            )

    setattr(wrapped, _SENTINEL, True)
    return wrapped


def _patch(cls: type, method_name: str, kind: str, attrs_fn: AttrsFn) -> None:
    """Wrap an async method defined *on this class* (not inherited)."""
    method = cls.__dict__.get(method_name)
    if method is None or getattr(method, _SENTINEL, False):
        return
    if not inspect.iscoroutinefunction(method):
        return
    setattr(cls, method_name, _make_wrapper(method, kind, attrs_fn))


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
        _patch(WorldModelRAG, "ingest", "graph.ingest", lambda self, a, k: {"graph": "world_model"})

    # -- Knowledge graph: property-graph backends --
    def property_graph() -> None:
        from ..retrieval import property_graph as pg

        for name in ("NetworkXPropertyGraphBackend", "Neo4jPropertyGraphBackend"):
            cls = getattr(pg, name, None)
            if cls is not None:
                _patch(cls, "search", "graph.query", _const(backend=name))
                _patch(cls, "ingest", "graph.ingest", _const(backend=name))

    # -- Knowledge mesh --
    def mesh() -> None:
        from ..mesh.core import KnowledgeMesh

        _patch(KnowledgeMesh, "query", "mesh.query", lambda self, a, k: {})
        _patch(KnowledgeMesh, "ingest_okf", "mesh.ingest", lambda self, a, k: {"format": "okf"})

    for step in (tools, memory, world_model, property_graph, mesh):
        _try(step)
