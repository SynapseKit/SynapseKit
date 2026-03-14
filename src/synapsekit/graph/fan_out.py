"""Parallel subgraph execution — fan-out/fan-in pattern."""

from __future__ import annotations

import asyncio
from typing import Any

from .node import NodeFn


def fan_out_node(
    subgraphs: list[Any],
    input_mappings: list[dict[str, str]] | None = None,
    output_key: str = "fan_out_results",
    merge_fn: Any | None = None,
) -> NodeFn:
    """Run multiple subgraphs in parallel and collect their results.

    Args:
        subgraphs: List of ``CompiledGraph`` instances to run concurrently.
        input_mappings: Optional per-subgraph input key mappings.
            Each dict maps ``{parent_key: sub_key}``.
            If ``None``, each subgraph receives the full parent state.
        output_key: State key to store the list of results.
        merge_fn: Optional function ``(list[dict]) -> dict`` to merge results
            into a single dict. If ``None``, results are stored as a list.

    Usage::

        fan = fan_out_node(
            subgraphs=[compiled_a, compiled_b, compiled_c],
            input_mappings=[
                {"query": "input"},
                {"query": "input"},
                {"query": "input"},
            ],
            output_key="results",
        )
        graph.add_node("parallel", fan)

    With a merge function::

        def merge(results):
            return {"combined": " | ".join(r.get("output", "") for r in results)}

        fan = fan_out_node(
            subgraphs=[sub_a, sub_b],
            merge_fn=merge,
        )
    """
    mappings = input_mappings or [None] * len(subgraphs)  # type: ignore[list-item]

    if len(mappings) != len(subgraphs):
        raise ValueError("input_mappings must have the same length as subgraphs.")

    async def _fn(state: dict[str, Any]) -> dict[str, Any]:
        tasks = []
        for sg, mapping in zip(subgraphs, mappings, strict=True):
            if mapping:
                sub_state = {sub_key: state[parent_key] for parent_key, sub_key in mapping.items()}
            else:
                sub_state = dict(state)
            tasks.append(sg.run(sub_state))

        results = await asyncio.gather(*tasks)

        if merge_fn is not None:
            merged: dict[str, Any] = merge_fn(list(results))
            return merged
        return {output_key: list(results)}

    return _fn
