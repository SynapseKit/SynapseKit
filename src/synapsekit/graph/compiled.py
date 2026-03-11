from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from .edge import ConditionalEdge, Edge
from .errors import GraphRuntimeError
from .mermaid import get_mermaid
from .state import END

if TYPE_CHECKING:
    from .checkpointers.base import BaseCheckpointer
    from .graph import StateGraph

_MAX_STEPS = 100


class CompiledGraph:
    """
    Runnable compiled graph produced by StateGraph.compile().
    Executes nodes wave by wave; parallel nodes in the same wave run concurrently.
    """

    def __init__(self, graph: StateGraph, max_steps: int | None = None) -> None:
        self._graph = graph
        self._max_steps = max_steps if max_steps is not None else _MAX_STEPS
        # Pre-build adjacency index for O(1) edge lookup per source node
        self._adj: dict[str, list[Edge | ConditionalEdge]] = {n: [] for n in graph._nodes}
        for edge in graph._edges:
            if edge.src in self._adj:
                self._adj[edge.src].append(edge)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def run(
        self,
        state: dict[str, Any],
        checkpointer: BaseCheckpointer | None = None,
        graph_id: str | None = None,
    ) -> dict[str, Any]:
        """Run the graph to completion and return the final state."""
        state = dict(state)
        async for _ in self._execute(state, checkpointer=checkpointer, graph_id=graph_id):
            pass
        return state

    async def stream(
        self,
        state: dict[str, Any],
        checkpointer: BaseCheckpointer | None = None,
        graph_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any]]:
        """
        Yield ``{"node": name, "state": snapshot}`` for each completed node.
        The caller receives incremental state updates as nodes finish.
        """
        state = dict(state)
        async for event in self._execute(state, checkpointer=checkpointer, graph_id=graph_id):
            yield event

    async def resume(
        self,
        graph_id: str,
        checkpointer: BaseCheckpointer,
    ) -> dict[str, Any]:
        """Resume execution from a checkpointed state."""
        saved = checkpointer.load(graph_id)
        if saved is None:
            raise GraphRuntimeError(f"No checkpoint found for graph_id={graph_id!r}.")
        _step, state = saved
        return await self.run(state, checkpointer=checkpointer, graph_id=graph_id)

    def run_sync(
        self,
        state: dict[str, Any],
        checkpointer: BaseCheckpointer | None = None,
        graph_id: str | None = None,
    ) -> dict[str, Any]:
        """Synchronous wrapper — works inside and outside a running event loop."""
        from .._compat import run_sync

        return run_sync(self.run(state, checkpointer=checkpointer, graph_id=graph_id))

    def get_mermaid(self) -> str:
        return get_mermaid(self._graph)

    # ------------------------------------------------------------------ #
    # Execution engine
    # ------------------------------------------------------------------ #

    async def _execute(
        self,
        state: dict[str, Any],
        checkpointer: BaseCheckpointer | None = None,
        graph_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any]]:
        graph = self._graph
        current_wave: list[str] = [graph._entry_point]  # type: ignore[list-item]
        steps = 0

        while current_wave:
            if steps >= self._max_steps:
                raise GraphRuntimeError(
                    f"Graph exceeded _MAX_STEPS={self._max_steps}. "
                    "Check for infinite loops in conditional edges."
                )
            steps += 1

            # Run all nodes in this wave concurrently
            results = await asyncio.gather(*[self._call_node(name, state) for name in current_wave])

            # Merge partial results into state and yield events
            for name, partial in zip(current_wave, results, strict=False):
                state.update(partial)
                yield {"node": name, "state": dict(state)}

            # Save checkpoint after wave completion
            if checkpointer is not None and graph_id is not None:
                checkpointer.save(graph_id, steps, dict(state))

            # Resolve next wave
            current_wave = await self._next_wave(current_wave, state)

    async def _call_node(self, name: str, state: dict[str, Any]) -> dict[str, Any]:
        node = self._graph._nodes.get(name)
        if node is None:
            raise GraphRuntimeError(f"Node {name!r} not found in graph.")
        result = node.fn(state)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, dict):
            raise GraphRuntimeError(
                f"Node {name!r} must return a dict, got {type(result).__name__!r}."
            )
        return result

    async def _next_wave(self, completed: list[str], state: dict[str, Any]) -> list[str]:
        """Determine which nodes to run next based on completed nodes and state."""
        next_nodes: list[str] = []
        seen: set[str] = set()

        for src in completed:
            for edge in self._adj.get(src, []):
                if isinstance(edge, Edge):
                    dst = edge.dst
                elif isinstance(edge, ConditionalEdge):
                    key = edge.condition_fn(state)
                    if inspect.isawaitable(key):
                        key = await key
                    dst = edge.mapping.get(str(key), END)
                else:
                    continue

                if dst == END:
                    continue
                if dst not in seen:
                    seen.add(dst)
                    next_nodes.append(dst)

        return next_nodes
