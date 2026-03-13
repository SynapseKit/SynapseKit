"""Subgraph support — nest a compiled graph as a node in a parent graph."""

from __future__ import annotations

from typing import Any

from .node import NodeFn


def subgraph_node(
    compiled_graph: Any,
    input_mapping: dict[str, str] | None = None,
    output_mapping: dict[str, str] | None = None,
) -> NodeFn:
    """Wrap a CompiledGraph as a node function for nesting in a parent graph.

    Args:
        compiled_graph: A ``CompiledGraph`` to run as a subgraph.
        input_mapping: Map parent state keys to subgraph state keys.
            E.g. ``{"parent_input": "input"}`` reads ``state["parent_input"]``
            and passes it as ``{"input": ...}`` to the subgraph.
        output_mapping: Map subgraph output keys to parent state keys.
            E.g. ``{"output": "sub_result"}`` takes the subgraph's ``"output"``
            and returns it as ``{"sub_result": ...}`` to the parent.

    Usage::

        # Build a subgraph
        sub = StateGraph()
        sub.add_node("process", process_fn)
        sub.set_entry_point("process").set_finish_point("process")
        compiled_sub = sub.compile()

        # Nest it in a parent graph
        parent = StateGraph()
        parent.add_node("sub", subgraph_node(
            compiled_sub,
            input_mapping={"query": "input"},
            output_mapping={"output": "sub_result"},
        ))
    """
    in_map = input_mapping or {}
    out_map = output_mapping or {}

    async def _fn(state: dict[str, Any]) -> dict[str, Any]:
        # Build subgraph initial state from parent state
        if in_map:
            sub_state = {sub_key: state[parent_key] for parent_key, sub_key in in_map.items()}
        else:
            sub_state = dict(state)

        # Run the subgraph
        result = await compiled_graph.run(sub_state)

        # Map subgraph output back to parent state keys
        if out_map:
            return {parent_key: result[sub_key] for sub_key, parent_key in out_map.items()}
        return result

    return _fn
