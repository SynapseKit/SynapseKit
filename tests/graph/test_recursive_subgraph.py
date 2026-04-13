"""Tests for recursive subgraph support via subgraph_node(StateGraph, ...).

Covers:
- Self-referential graph terminates on a base case via conditional edge
- RecursionDepthError raised when max_recursion_depth is exceeded
- Custom max_recursion_depth is respected
- __recursion_depth__ key is NOT present in the final parent output
- Depth increments correctly across recursive calls
- Passing an uncompiled StateGraph (lazy compile) works correctly
- input_mapping / output_mapping compose with recursion
- on_error=skip composes with recursion depth exceeded
- max_recursion_depth=0 raises ValueError at factory time
- End-to-end: recursive countdown (divide-and-conquer style)
"""

from __future__ import annotations

import pytest

from synapsekit.graph.errors import RecursionDepthError
from synapsekit.graph.graph import StateGraph
from synapsekit.graph.state import END
from synapsekit.graph.subgraph import _RECURSION_DEPTH_KEY, subgraph_node

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _simple_compiled_graph(output: dict) -> object:
    """Return a compiled subgraph that always succeeds with *output*."""

    async def ok_node(state):
        return output

    g = StateGraph()
    g.add_node("ok", ok_node)
    g.set_entry_point("ok").set_finish_point("ok")
    return g.compile()


# ------------------------------------------------------------------ #
# Validation
# ------------------------------------------------------------------ #


def test_subgraph_node_returns_coroutine_function():
    import inspect

    sub = _simple_compiled_graph({})
    node_fn = subgraph_node(sub)
    assert inspect.iscoroutinefunction(node_fn)


def test_invalid_max_recursion_depth_zero_raises():
    sub = _simple_compiled_graph({})
    with pytest.raises(ValueError, match="max_recursion_depth must be >= 1"):
        subgraph_node(sub, max_recursion_depth=0)


def test_invalid_max_recursion_depth_negative_raises():
    sub = _simple_compiled_graph({})
    with pytest.raises(ValueError, match="max_recursion_depth must be >= 1"):
        subgraph_node(sub, max_recursion_depth=-3)


# ------------------------------------------------------------------ #
# Non-recursive usage — existing behaviour unchanged
# ------------------------------------------------------------------ #


async def test_non_recursive_compiled_graph_unaffected():
    """A regular non-recursive subgraph_node(CompiledGraph) still works."""
    sub = _simple_compiled_graph({"answer": 42})
    node_fn = subgraph_node(sub)  # CompiledGraph, no recursion

    result = await node_fn({})
    assert result == {"answer": 42}
    assert _RECURSION_DEPTH_KEY not in result


async def test_depth_key_not_in_output_for_non_recursive():
    """__recursion_depth__ must never leak into the parent state."""
    sub = _simple_compiled_graph({"x": 1})
    node_fn = subgraph_node(sub)

    # Even if depth key is in parent state it should be stripped from output
    result = await node_fn({_RECURSION_DEPTH_KEY: 2})
    assert _RECURSION_DEPTH_KEY not in result


# ------------------------------------------------------------------ #
# RecursionDepthError guard
# ------------------------------------------------------------------ #


async def test_depth_guard_raises_when_limit_reached():
    """Calling _fn with depth already at the limit must raise RecursionDepthError."""
    sub = _simple_compiled_graph({"ok": True})
    node_fn = subgraph_node(sub, max_recursion_depth=3)

    # Simulate state that is already AT the limit
    with pytest.raises(RecursionDepthError, match="max_recursion_depth=3"):
        await node_fn({_RECURSION_DEPTH_KEY: 3})


async def test_depth_guard_allows_call_below_limit():
    """Should succeed when current depth is still below the limit."""
    sub = _simple_compiled_graph({"ok": True})
    node_fn = subgraph_node(sub, max_recursion_depth=3)

    result = await node_fn({_RECURSION_DEPTH_KEY: 2})
    assert result["ok"] is True


async def test_custom_max_recursion_depth_respected():
    sub = _simple_compiled_graph({})
    node_fn = subgraph_node(sub, max_recursion_depth=1)

    # depth=0 → OK (0 < 1)
    await node_fn({})

    # depth=1 → EXCEEDS
    with pytest.raises(RecursionDepthError):
        await node_fn({_RECURSION_DEPTH_KEY: 1})


# ------------------------------------------------------------------ #
# RecursionDepthError is a subclass of GraphRuntimeError
# ------------------------------------------------------------------ #


def test_recursion_depth_error_is_graph_runtime_error():
    from synapsekit.graph.errors import GraphRuntimeError

    err = RecursionDepthError("oops")
    assert isinstance(err, GraphRuntimeError)


# ------------------------------------------------------------------ #
# Lazy compile — passing a StateGraph (self-referential)
# ------------------------------------------------------------------ #


async def test_stategraph_lazy_compile_basic():
    """subgraph_node(StateGraph) lazy-compiles and runs correctly."""

    # Simple graph: one node that doubles a counter, exits immediately
    async def double(state):
        return {"value": state.get("value", 0) * 2}

    sg = StateGraph()
    sg.add_node("double", double)
    sg.set_entry_point("double").set_finish_point("double")

    # Pass StateGraph (not compiled) — it should compile lazily on first call
    node_fn = subgraph_node(sg)
    result = await node_fn({"value": 5})
    assert result["value"] == 10


async def test_stategraph_lazy_compile_called_twice():
    """Lazy compile must happen only once — second call reuses the compiled graph."""

    compile_calls = {"n": 0}
    original_compile = StateGraph.compile

    def tracking_compile(self, **kwargs):
        compile_calls["n"] += 1
        return original_compile(self, **kwargs)

    sg = StateGraph()

    async def noop(state):
        return {}

    sg.add_node("n", noop)
    sg.set_entry_point("n").set_finish_point("n")

    node_fn = subgraph_node(sg)

    # Patch compile temporarily so we can count calls
    StateGraph.compile = tracking_compile
    try:
        await node_fn({})
        await node_fn({})
    finally:
        StateGraph.compile = original_compile

    # compile() must have been called exactly once (lazy init)
    assert compile_calls["n"] == 1


# ------------------------------------------------------------------ #
# Self-referential recursive graph — terminates via conditional edge
# ------------------------------------------------------------------ #


async def test_recursive_graph_countdown():
    """Recursive graph decrements a counter until 0, then exits."""

    # Each recursive call decrements `n` by 1.
    # When n <= 0 the conditional edge routes to END.

    graph = StateGraph()

    async def decrement(state):
        return {"n": state["n"] - 1}

    def route(state):
        return "recurse" if state["n"] > 0 else END

    # Register the recursive node BEFORE compile — using the uncompiled StateGraph
    graph.add_node("decrement", decrement)
    graph.add_node("recurse", subgraph_node(graph, max_recursion_depth=20))
    graph.add_conditional_edge("decrement", route, {"recurse": "recurse", END: END})
    graph.add_edge("recurse", END)
    graph.set_entry_point("decrement")

    compiled = graph.compile(allow_cycles=True)
    result = await compiled.run({"n": 5})

    assert result["n"] == 0
    assert _RECURSION_DEPTH_KEY not in result


async def test_recursive_graph_accumulates_result():
    """Recursive graph sums items one by one (divide-and-conquer style)."""

    graph = StateGraph()

    async def process(state):
        items = list(state.get("items", []))
        total = state.get("total", 0)
        if not items:
            return {"items": [], "total": total}
        head, *tail = items
        return {"items": tail, "total": total + head}

    def should_recurse(state):
        return "recurse" if state["items"] else END

    graph.add_node("process", process)
    graph.add_node("recurse", subgraph_node(graph, max_recursion_depth=20))
    graph.add_conditional_edge("process", should_recurse, {"recurse": "recurse", END: END})
    graph.add_edge("recurse", END)
    graph.set_entry_point("process")

    compiled = graph.compile(allow_cycles=True)
    result = await compiled.run({"items": [1, 2, 3, 4, 5], "total": 0})

    assert result["total"] == 15  # 1+2+3+4+5
    assert result["items"] == []
    assert _RECURSION_DEPTH_KEY not in result


async def test_recursive_graph_hits_depth_limit():
    """Recursive graph that never exits must hit RecursionDepthError."""

    graph = StateGraph()

    async def noop(state):
        return {}

    # Always recurse — never exit via base case
    graph.add_node("noop", noop)
    graph.add_node("recurse", subgraph_node(graph, max_recursion_depth=5))
    graph.add_edge("noop", "recurse")
    graph.add_edge("recurse", END)
    graph.set_entry_point("noop")

    compiled = graph.compile(allow_cycles=True)

    with pytest.raises(RecursionDepthError, match="max_recursion_depth=5"):
        await compiled.run({})


# ------------------------------------------------------------------ #
# input_mapping / output_mapping compose with recursion
# ------------------------------------------------------------------ #


async def test_recursive_with_input_output_mapping():
    """input_mapping and output_mapping work correctly alongside recursion."""

    # Subgraph receives "val", returns "out"
    async def double_node(state):
        return {"out": state["val"] * 2}

    sg = StateGraph()
    sg.add_node("double", double_node)
    sg.set_entry_point("double").set_finish_point("double")

    node_fn = subgraph_node(
        sg,
        input_mapping={"number": "val"},
        output_mapping={"out": "result"},
        max_recursion_depth=5,
    )

    result = await node_fn({"number": 7})
    assert result["result"] == 14
    assert _RECURSION_DEPTH_KEY not in result


# ------------------------------------------------------------------ #
# on_error=skip composes with RecursionDepthError
# ------------------------------------------------------------------ #


async def test_recursion_depth_error_with_on_error_skip():
    """When on_error='skip', a RecursionDepthError is caught and reported."""
    sub = _simple_compiled_graph({})
    node_fn = subgraph_node(sub, on_error="skip", max_recursion_depth=2)

    # Trigger depth limit
    result = await node_fn({_RECURSION_DEPTH_KEY: 2})

    assert "__subgraph_error__" in result
    err = result["__subgraph_error__"]
    assert err["type"] == "RecursionDepthError"
    assert "max_recursion_depth=2" in err["message"]


# ------------------------------------------------------------------ #
# Top-level import
# ------------------------------------------------------------------ #


def test_recursion_depth_error_importable_from_synapsekit():
    from synapsekit import RecursionDepthError as RDE

    assert RDE is RecursionDepthError


def test_recursion_depth_error_importable_from_graph():
    from synapsekit.graph import RecursionDepthError as RDE

    assert RDE is RecursionDepthError
