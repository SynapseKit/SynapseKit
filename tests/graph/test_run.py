"""Tests for CompiledGraph.run() — linear, conditional, parallel."""

import pytest

from synapsekit.graph.errors import GraphRuntimeError
from synapsekit.graph.graph import StateGraph
from synapsekit.graph.state import END

# ------------------------------------------------------------------ #
# Linear pipeline
# ------------------------------------------------------------------ #


async def test_linear_two_nodes():
    async def add_greeting(state):
        return {"greeting": f"Hello, {state['name']}"}

    async def add_exclaim(state):
        return {"result": state["greeting"] + "!"}

    g = StateGraph()
    g.add_node("greet", add_greeting).add_node("exclaim", add_exclaim)
    g.add_edge("greet", "exclaim")
    g.set_entry_point("greet").set_finish_point("exclaim")

    result = await g.compile().run({"name": "World"})
    assert result["result"] == "Hello, World!"


async def test_single_node_with_finish():
    async def double(state):
        return {"value": state["x"] * 2}

    g = StateGraph()
    g.add_node("double", double)
    g.set_entry_point("double").set_finish_point("double")

    result = await g.compile().run({"x": 5})
    assert result["value"] == 10


async def test_state_is_not_mutated_by_caller():
    """Original dict passed by caller should not be modified."""

    async def add_key(state):
        return {"extra": True}

    g = StateGraph()
    g.add_node("node", add_key)
    g.set_entry_point("node").set_finish_point("node")

    original = {"input": "hello"}
    await g.compile().run(original)
    assert "extra" not in original


# ------------------------------------------------------------------ #
# Conditional routing (sync route fn)
# ------------------------------------------------------------------ #


async def test_conditional_sync_route_left():
    async def classify(state):
        return {"label": "even" if state["n"] % 2 == 0 else "odd"}

    async def even_handler(state):
        return {"result": "even path"}

    async def odd_handler(state):
        return {"result": "odd path"}

    def route(state):
        return state["label"]

    g = StateGraph()
    g.add_node("classify", classify)
    g.add_node("even", even_handler)
    g.add_node("odd", odd_handler)
    g.add_conditional_edge("classify", route, {"even": "even", "odd": "odd"})
    g.add_edge("even", END)
    g.add_edge("odd", END)
    g.set_entry_point("classify")

    result_even = await g.compile().run({"n": 4})
    assert result_even["result"] == "even path"

    result_odd = await g.compile().run({"n": 3})
    assert result_odd["result"] == "odd path"


async def test_conditional_async_route():
    async def step(state):
        return {"value": state["x"] + 1}

    async def async_route(state):
        return "big" if state["value"] > 5 else "small"

    async def big_node(state):
        return {"tag": "BIG"}

    async def small_node(state):
        return {"tag": "SMALL"}

    g = StateGraph()
    g.add_node("step", step)
    g.add_node("big", big_node)
    g.add_node("small", small_node)
    g.add_conditional_edge("step", async_route, {"big": "big", "small": "small"})
    g.add_edge("big", END).add_edge("small", END)
    g.set_entry_point("step")

    r1 = await g.compile().run({"x": 10})
    assert r1["tag"] == "BIG"

    r2 = await g.compile().run({"x": 1})
    assert r2["tag"] == "SMALL"


# ------------------------------------------------------------------ #
# Parallel nodes (same wave)
# ------------------------------------------------------------------ #


async def test_parallel_nodes():
    """Two nodes with no order dependency should both execute."""

    async def fetch_a(state):
        return {"a": "result_a"}

    async def fetch_b(state):
        return {"b": "result_b"}

    async def merge(state):
        return {"combined": f"{state['a']}+{state['b']}"}

    g = StateGraph()
    g.add_node("fetch_a", fetch_a)
    g.add_node("fetch_b", fetch_b)
    g.add_node("merge", merge)

    # Both fetch_* nodes are triggered from start via separate edges from a fanout node
    async def fanout(state):
        return {}

    g.add_node("fanout", fanout)
    g.add_edge("fanout", "fetch_a")
    g.add_edge("fanout", "fetch_b")
    g.add_edge("fetch_a", "merge")
    g.add_edge("fetch_b", "merge")
    g.set_entry_point("fanout").set_finish_point("merge")

    result = await g.compile().run({})
    assert result["a"] == "result_a"
    assert result["b"] == "result_b"
    assert result["combined"] == "result_a+result_b"


# ------------------------------------------------------------------ #
# Sync node function
# ------------------------------------------------------------------ #


async def test_sync_node_fn():
    def sync_fn(state):
        return {"doubled": state["x"] * 2}

    g = StateGraph()
    g.add_node("double", sync_fn)
    g.set_entry_point("double").set_finish_point("double")

    result = await g.compile().run({"x": 7})
    assert result["doubled"] == 14


# ------------------------------------------------------------------ #
# Error cases
# ------------------------------------------------------------------ #


async def test_node_returning_non_dict_raises():
    def bad_fn(state):
        return "not a dict"

    g = StateGraph()
    g.add_node("bad", bad_fn)
    g.set_entry_point("bad").set_finish_point("bad")

    with pytest.raises(GraphRuntimeError, match="must return a dict"):
        await g.compile().run({})


async def test_max_steps_guard():
    """Graph with always-looping conditional should hit _MAX_STEPS."""
    from synapsekit.graph.compiled import _MAX_STEPS

    async def loop_node(state):
        return {"count": state.get("count", 0) + 1}

    # Bypass cycle detection (conditional edges are not checked statically)
    def always_loop(state):
        return "loop"

    g = StateGraph()
    g.add_node("loop", loop_node)
    g.add_conditional_edge("loop", always_loop, {"loop": "loop"})
    g.set_entry_point("loop")

    with pytest.raises(GraphRuntimeError, match=f"_MAX_STEPS={_MAX_STEPS}"):
        await g.compile().run({})
