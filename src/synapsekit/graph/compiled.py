from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from .edge import ConditionalEdge, Edge
from .errors import GraphRuntimeError
from .interrupt import GraphInterrupt
from .mermaid import get_mermaid
from .state import END
from .streaming import EventHooks, GraphEvent

if TYPE_CHECKING:
    from .checkpointers.base import BaseCheckpointer
    from .graph import StateGraph

_MAX_STEPS = 100
_CHECKPOINT_VERSION_KEY = "__synapsekit_graph_version"


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

    def __repr__(self) -> str:
        nodes = len(self._graph._nodes)
        edges = len(self._graph._edges)
        return f"CompiledGraph(nodes={nodes}, edges={edges}, max_steps={self._max_steps})"

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def _merge_state(self, state: dict[str, Any], partial: dict[str, Any]) -> None:
        """Merge partial state into current state, using reducers if available."""
        schema = self._graph._state_schema
        if schema is not None:
            schema.merge(state, partial)
        else:
            state.update(partial)

    async def run(
        self,
        state: dict[str, Any],
        checkpointer: BaseCheckpointer | None = None,
        graph_id: str | None = None,
        hooks: EventHooks | None = None,
    ) -> dict[str, Any]:
        """Run the graph to completion and return the final state."""
        state = dict(state)
        async for _ in self._execute(
            state, checkpointer=checkpointer, graph_id=graph_id, hooks=hooks
        ):
            pass
        return state

    async def stream(
        self,
        state: dict[str, Any],
        checkpointer: BaseCheckpointer | None = None,
        graph_id: str | None = None,
        hooks: EventHooks | None = None,
    ) -> AsyncGenerator[dict[str, Any]]:
        """
        Yield ``{"node": name, "state": snapshot}`` for each completed node.
        The caller receives incremental state updates as nodes finish.
        """
        state = dict(state)
        async for event in self._execute(
            state, checkpointer=checkpointer, graph_id=graph_id, hooks=hooks
        ):
            yield event

    async def resume(
        self,
        graph_id: str,
        checkpointer: BaseCheckpointer,
        updates: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Resume execution from a checkpointed state.

        Args:
            graph_id: The graph execution ID to resume.
            checkpointer: The checkpointer that holds the saved state.
            updates: Optional state updates to apply before resuming
                (e.g. human-provided edits after a ``GraphInterrupt``).
        """
        saved = checkpointer.load(graph_id)
        if saved is None:
            raise GraphRuntimeError(f"No checkpoint found for graph_id={graph_id!r}.")
        _step, state = saved

        checkpoint_version = str(state.pop(_CHECKPOINT_VERSION_KEY, "1"))
        if checkpoint_version != self._graph.version:
            state = await self._apply_migration_chain(
                state=state,
                from_version=checkpoint_version,
                to_version=self._graph.version,
            )

        if updates:
            state.update(updates)
        return await self.run(state, checkpointer=checkpointer, graph_id=graph_id)

    async def _apply_migration_chain(
        self,
        *,
        state: dict[str, Any],
        from_version: str,
        to_version: str,
    ) -> dict[str, Any]:
        current_version = from_version
        current_state = dict(state)
        seen: set[str] = set()

        while current_version != to_version:
            if current_version in seen:
                raise GraphRuntimeError(
                    f"Detected migration cycle while upgrading graph state from "
                    f"{from_version!r} to {to_version!r}."
                )
            seen.add(current_version)

            migration = self._graph.migrations.get(current_version)
            if migration is None:
                raise GraphRuntimeError(
                    f"No migration path from graph version {current_version!r} to {to_version!r}."
                )

            migrated = migration(dict(current_state))
            if inspect.isawaitable(migrated):
                migrated = await migrated

            if isinstance(migrated, tuple) and len(migrated) == 2:
                next_version, next_state = migrated
                current_version = str(next_version)
                current_state = dict(next_state)
                continue

            if isinstance(migrated, dict):
                # Single-step migration from current -> target.
                current_state = dict(migrated)
                current_version = to_version
                continue

            raise GraphRuntimeError(
                "Migration functions must return either a state dict "
                "or a (next_version, state_dict) tuple."
            )

        return current_state

    def run_sync(
        self,
        state: dict[str, Any],
        checkpointer: BaseCheckpointer | None = None,
        graph_id: str | None = None,
        hooks: EventHooks | None = None,
    ) -> dict[str, Any]:
        """Synchronous wrapper — works inside and outside a running event loop."""
        from .._compat import run_sync

        return run_sync(self.run(state, checkpointer=checkpointer, graph_id=graph_id, hooks=hooks))

    async def stream_tokens(
        self,
        state: dict[str, Any],
    ) -> AsyncGenerator[dict[str, Any]]:
        """Yield token-level events from LLM nodes.

        Yields dicts with either:
        - ``{"type": "token", "node": name, "token": str}`` for streaming tokens
        - ``{"type": "node_complete", "node": name, "state": dict}`` for non-streaming nodes

        LLM nodes are detected by checking if the node function's return dict
        contains a ``"__stream__"`` key with an async generator.
        """
        state = dict(state)
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

            for name in current_wave:
                node = graph._nodes.get(name)
                if node is None:
                    raise GraphRuntimeError(f"Node {name!r} not found in graph.")

                result = node.fn(state)
                if inspect.isawaitable(result):
                    result = await result

                if not isinstance(result, dict):
                    raise GraphRuntimeError(
                        f"Node {name!r} must return a dict, got {type(result).__name__!r}."
                    )

                # Check for streaming token generator
                stream_gen = result.pop("__stream__", None)
                if stream_gen is not None:
                    collected: list[str] = []
                    async for token in stream_gen:
                        collected.append(token)
                        yield {"type": "token", "node": name, "token": token}
                    # Store the full text in the result
                    if "__stream_key__" in result:
                        result[result.pop("__stream_key__")] = "".join(collected)

                self._merge_state(state, result)
                yield {"type": "node_complete", "node": name, "state": dict(state)}

            current_wave = await self._next_wave(current_wave, state)

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
        hooks: EventHooks | None = None,
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

            # Emit wave_start event
            if hooks is not None:
                await hooks.emit(
                    GraphEvent(
                        event_type="wave_start",
                        data={"wave": current_wave, "step": steps},
                    )
                )

            # Emit node_start events
            if hooks is not None:
                for name in current_wave:
                    await hooks.emit(GraphEvent(event_type="node_start", node=name))

            # Run all nodes in this wave concurrently
            try:
                results = await asyncio.gather(
                    *[self._call_node(name, state) for name in current_wave]
                )
            except GraphInterrupt as exc:
                # Save state and raise InterruptState for the caller
                if checkpointer is not None and graph_id is not None:
                    checkpointer.save(graph_id, steps, self._checkpoint_state(state))
                raise GraphInterrupt(exc.message, exc.data) from None

            # Merge partial results into state and yield events
            for name, partial in zip(current_wave, results, strict=False):
                self._merge_state(state, partial)
                yield {"node": name, "state": dict(state)}

                # Emit node_complete event
                if hooks is not None:
                    await hooks.emit(
                        GraphEvent(
                            event_type="node_complete",
                            node=name,
                            state=dict(state),
                        )
                    )

            # Emit wave_complete event
            if hooks is not None:
                await hooks.emit(
                    GraphEvent(
                        event_type="wave_complete",
                        data={"wave": current_wave, "step": steps},
                        state=dict(state),
                    )
                )

            # Save checkpoint after wave completion
            if checkpointer is not None and graph_id is not None:
                checkpointer.save(graph_id, steps, self._checkpoint_state(state))

            # Resolve next wave
            current_wave = await self._next_wave(current_wave, state)

    def _checkpoint_state(self, state: dict[str, Any]) -> dict[str, Any]:
        payload = dict(state)
        payload[_CHECKPOINT_VERSION_KEY] = self._graph.version
        return payload

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
