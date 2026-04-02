"""Subgraph support — nest a compiled graph as a node in a parent graph."""

from __future__ import annotations

from typing import Any, Literal

from .node import NodeFn


def subgraph_node(
    compiled_graph: Any,
    input_mapping: dict[str, str] | None = None,
    output_mapping: dict[str, str] | None = None,
    *,
    on_error: Literal["raise", "retry", "fallback", "skip"] = "raise",
    max_retries: int = 3,
    fallback: Any | None = None,
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
        on_error: How to handle subgraph failures:
            - ``"raise"``    — re-raise the exception (default).
            - ``"retry"``    — re-run up to *max_retries* times, then raise.
            - ``"fallback"`` — run *fallback* (another CompiledGraph) on failure.
            - ``"skip"``     — return an empty dict and continue the parent graph.
        max_retries: Maximum number of attempts when *on_error* is ``"retry"``.
            Must be >= 1. Ignored for other strategies.
        fallback: A ``CompiledGraph`` used when *on_error* is ``"fallback"``.
            Required when *on_error="fallback"*; ignored otherwise.

    The parent state will contain ``"__subgraph_error__"`` after a handled
    failure (``"retry"`` exhausted, ``"fallback"`` used, or ``"skip"``), set to
    a dict with keys ``"type"``, ``"message"``, and ``"attempts"``.

    Usage::

        # Basic nesting (raises on error)
        parent.add_node("sub", subgraph_node(compiled_sub))

        # Retry up to 5 times before raising
        parent.add_node("sub", subgraph_node(compiled_sub, on_error="retry", max_retries=5))

        # Fall back to a simpler subgraph on failure
        parent.add_node(
            "sub",
            subgraph_node(compiled_sub, on_error="fallback", fallback=fallback_sub),
        )

        # Silently skip the subgraph on failure and continue
        parent.add_node("sub", subgraph_node(compiled_sub, on_error="skip"))
    """
    if on_error == "fallback" and fallback is None:
        raise ValueError("on_error='fallback' requires a fallback CompiledGraph.")
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1.")

    in_map = input_mapping or {}
    out_map = output_mapping or {}

    def _build_sub_state(state: dict[str, Any]) -> dict[str, Any]:
        if in_map:
            return {sub_key: state[parent_key] for parent_key, sub_key in in_map.items()}
        return dict(state)

    def _map_output(result: dict[str, Any]) -> dict[str, Any]:
        if out_map:
            return {parent_key: result[sub_key] for sub_key, parent_key in out_map.items()}
        return dict(result)

    async def _fn(state: dict[str, Any]) -> dict[str, Any]:
        sub_state = _build_sub_state(state)
        last_exc: BaseException | None = None

        # ── "raise" (fast path — no error wrapping overhead) ──────────────────
        if on_error == "raise":
            result = await compiled_graph.run(sub_state)
            return _map_output(result)

        # ── "retry" ────────────────────────────────────────────────────────────
        if on_error == "retry":
            for _ in range(max_retries):
                try:
                    result = await compiled_graph.run(sub_state)
                    return _map_output(result)
                except Exception as exc:
                    last_exc = exc
            # All attempts exhausted — surface error info and re-raise
            assert last_exc is not None
            exc_type = type(last_exc)
            raise exc_type(
                f"Subgraph failed after {max_retries} attempt(s): {last_exc}"
            ) from last_exc

        # ── "fallback" ─────────────────────────────────────────────────────────
        if on_error == "fallback":
            try:
                result = await compiled_graph.run(sub_state)
                return _map_output(result)
            except Exception as exc:
                last_exc = exc
                fallback_result = await fallback.run(sub_state)  # type: ignore[union-attr]
                mapped = _map_output(fallback_result)
                mapped["__subgraph_error__"] = {
                    "type": type(last_exc).__name__,
                    "message": str(last_exc),
                    "attempts": 1,
                }
                return mapped

        # ── "skip" ─────────────────────────────────────────────────────────────
        if on_error == "skip":
            try:
                result = await compiled_graph.run(sub_state)
                return _map_output(result)
            except Exception as exc:
                return {
                    "__subgraph_error__": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "attempts": 1,
                    }
                }

        # Unreachable — guard against bad Literal values at runtime
        raise ValueError(f"Unknown on_error strategy: {on_error!r}")  # pragma: no cover

    return _fn
