from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING, Any, AsyncGenerator

from .edge import ConditionalEdge, Edge
from .errors import GraphRuntimeError
from .mermaid import get_mermaid
from .state import END

if TYPE_CHECKING:
    from .graph import StateGraph

_MAX_STEPS = 100


class CompiledGraph:
    """
    Runnable compiled graph produced by StateGraph.compile().
    Executes nodes wave by wave; parallel nodes in the same wave run concurrently.
    """

    def __init__(self, graph: "StateGraph") -> None:
        self._graph = graph

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Run the graph to completion and return the final state."""
        state = dict(state)
        async for _ in self._execute(state):
            pass
        return state

    async def stream(self, state: dict[str, Any]) -> AsyncGenerator[dict[str, Any], None]:
        """
        Yield ``{"node": name, "state": snapshot}`` for each completed node.
        The caller receives incremental state updates as nodes finish.
        """
        state = dict(state)
        async for event in self._execute(state):
            yield event

    def run_sync(self, state: dict[str, Any]) -> dict[str, Any]:
        """Synchronous wrapper — works inside and outside a running event loop."""
        from .._compat import run_sync
        return run_sync(self.run(state))

    def get_mermaid(self) -> str:
        return get_mermaid(self._graph)

    # ------------------------------------------------------------------ #
    # Execution engine
    # ------------------------------------------------------------------ #

    async def _execute(
        self, state: dict[str, Any]
    ) -> AsyncGenerator[dict[str, Any], None]:
        graph = self._graph
        current_wave: list[str] = [graph._entry_point]  # type: ignore[list-item]
        steps = 0

        while current_wave:
            if steps >= _MAX_STEPS:
                raise GraphRuntimeError(
                    f"Graph exceeded _MAX_STEPS={_MAX_STEPS}. "
                    "Check for infinite loops in conditional edges."
                )
            steps += 1

            # Run all nodes in this wave concurrently
            results = await asyncio.gather(
                *[self._call_node(name, state) for name in current_wave]
            )

            # Merge partial results into state and yield events
            for name, partial in zip(current_wave, results):
                state.update(partial)
                yield {"node": name, "state": dict(state)}

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

    async def _next_wave(
        self, completed: list[str], state: dict[str, Any]
    ) -> list[str]:
        """Determine which nodes to run next based on completed nodes and state."""
        next_nodes: list[str] = []
        seen: set[str] = set()

        for src in completed:
            for edge in self._graph._edges:
                if edge.src != src:
                    continue
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
