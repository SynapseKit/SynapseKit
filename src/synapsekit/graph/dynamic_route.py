"""Dynamic route node — route to subgraphs at runtime based on state."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from .node import NodeFn


def dynamic_route_node(
    routing_fn: Callable[[dict[str, Any]], str | Any],
    subgraphs: dict[str, Any],
    input_mapping: dict[str, str] | None = None,
    output_mapping: dict[str, str] | None = None,
) -> NodeFn:
    """Factory that returns a node function routing to subgraphs dynamically.

    ``routing_fn(state)`` returns a key that selects which compiled subgraph
    from *subgraphs* to execute.  Supports both sync and async routing
    functions.

    Args:
        routing_fn: Callable ``(state) -> key`` returning a key into
            *subgraphs*.  May be sync or async.
        subgraphs: Mapping of route keys to :class:`CompiledGraph` instances.
        input_mapping: Map parent state keys to subgraph state keys (same
            semantics as :func:`subgraph_node`).
        output_mapping: Map subgraph output keys back to parent state keys.

    Usage::

        def route(state):
            return "fast" if state.get("urgent") else "thorough"

        graph.add_node("router", dynamic_route_node(
            routing_fn=route,
            subgraphs={"fast": fast_graph, "thorough": thorough_graph},
        ))
    """
    in_map = input_mapping or {}
    out_map = output_mapping or {}

    async def _fn(state: dict[str, Any]) -> dict[str, Any]:
        # Resolve route key (sync or async routing_fn)
        result = routing_fn(state)
        if inspect.isawaitable(result):
            result = await result
        route_key = result

        subgraph = subgraphs.get(route_key)
        if subgraph is None:
            raise ValueError(
                f"Unknown route key: {route_key!r}. "
                f"Available routes: {', '.join(subgraphs)}"
            )

        # Build subgraph initial state from parent state
        if in_map:
            sub_state = {sub_key: state[parent_key] for parent_key, sub_key in in_map.items()}
        else:
            sub_state = dict(state)

        # Run the selected subgraph
        sub_result = await subgraph.run(sub_state)

        # Map subgraph output back to parent state keys
        if out_map:
            return {parent_key: sub_result[sub_key] for sub_key, parent_key in out_map.items()}
        return dict(sub_result)

    return _fn
