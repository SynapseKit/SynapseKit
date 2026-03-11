"""Tests for CompiledGraph.stream() and run_sync()."""
import pytest
from synapsekit.graph.graph import StateGraph
from synapsekit.graph.state import END


async def _inc(state):
    return {"count": state.get("count", 0) + 1}


async def _double(state):
    return {"value": state["x"] * 2}


# ------------------------------------------------------------------ #
# stream() yields events
# ------------------------------------------------------------------ #

async def test_stream_yields_node_events():
    g = StateGraph()
    g.add_node("a", _inc).add_node("b", _inc)
    g.add_edge("a", "b")
    g.set_entry_point("a").set_finish_point("b")

    events = []
    async for event in g.compile().stream({"count": 0}):
        events.append(event)

    assert len(events) == 2
    assert events[0]["node"] == "a"
    assert events[1]["node"] == "b"


async def test_stream_event_has_state_key():
    g = StateGraph()
    g.add_node("double", _double)
    g.set_entry_point("double").set_finish_point("double")

    events = []
    async for event in g.compile().stream({"x": 3}):
        events.append(event)

    assert len(events) == 1
    assert "state" in events[0]
    assert events[0]["state"]["value"] == 6


async def test_stream_state_accumulates():
    async def add_a(state):
        return {"a": 1}

    async def add_b(state):
        return {"b": 2}

    g = StateGraph()
    g.add_node("a", add_a).add_node("b", add_b)
    g.add_edge("a", "b")
    g.set_entry_point("a").set_finish_point("b")

    events = []
    async for event in g.compile().stream({}):
        events.append(event)

    # After node "a": state has "a"
    assert events[0]["state"].get("a") == 1
    # After node "b": state has both "a" and "b"
    assert events[1]["state"].get("a") == 1
    assert events[1]["state"].get("b") == 2


async def test_stream_final_state_matches_run():
    async def process(state):
        return {"result": state["x"] ** 2}

    g = StateGraph()
    g.add_node("sq", process)
    g.set_entry_point("sq").set_finish_point("sq")

    compiled = g.compile()
    run_result = await compiled.run({"x": 5})

    events = []
    async for event in compiled.stream({"x": 5}):
        events.append(event)

    assert events[-1]["state"]["result"] == run_result["result"]


async def test_stream_single_node():
    g = StateGraph()
    g.add_node("only", _inc)
    g.set_entry_point("only").set_finish_point("only")

    events = [e async for e in g.compile().stream({})]
    assert len(events) == 1
    assert events[0]["node"] == "only"


# ------------------------------------------------------------------ #
# run_sync()
# ------------------------------------------------------------------ #

def test_run_sync_basic():
    async def sq(state):
        return {"result": state["x"] ** 2}

    g = StateGraph()
    g.add_node("sq", sq)
    g.set_entry_point("sq").set_finish_point("sq")

    result = g.compile().run_sync({"x": 4})
    assert result["result"] == 16


def test_run_sync_linear_chain():
    async def step1(state):
        return {"a": state["x"] + 1}

    async def step2(state):
        return {"b": state["a"] * 2}

    g = StateGraph()
    g.add_node("s1", step1).add_node("s2", step2)
    g.add_edge("s1", "s2")
    g.set_entry_point("s1").set_finish_point("s2")

    result = g.compile().run_sync({"x": 3})
    assert result["a"] == 4
    assert result["b"] == 8
