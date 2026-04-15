"""End-to-end Graph workflow tests.

Tests full compile → run cycles for StateGraph with no API calls.
"""
from __future__ import annotations

import pytest

from synapsekit import StateGraph
from synapsekit.graph.errors import GraphConfigError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _upper(state: dict) -> dict:
    return {"text": state.get("text", "").upper()}


async def _reverse(state: dict) -> dict:
    return {"text": state.get("text", "")[::-1]}


async def _prefix(state: dict) -> dict:
    return {"text": "PREFIX_" + state.get("text", "")}


# ---------------------------------------------------------------------------
# 1. Linear 2-node graph
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_linear_graph_run():
    g = StateGraph()
    g.add_node("upper", _upper)
    g.add_node("reverse", _reverse)
    g.add_edge("upper", "reverse")
    g.set_entry_point("upper")
    g.set_finish_point("reverse")

    compiled = g.compile()
    result = await compiled.run({"text": "hello"})
    # "hello" → upper → "HELLO" → reverse → "OLLEH"
    assert result["text"] == "OLLEH"


# ---------------------------------------------------------------------------
# 2. Single-node graph
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_node_graph():
    g = StateGraph()
    g.add_node("upper", _upper)
    g.set_entry_point("upper")
    g.set_finish_point("upper")

    compiled = g.compile()
    result = await compiled.run({"text": "test"})
    assert result["text"] == "TEST"


# ---------------------------------------------------------------------------
# 3. Three-node linear chain
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_three_node_chain():
    g = StateGraph()
    g.add_node("a", _upper)
    g.add_node("b", _reverse)
    g.add_node("c", _prefix)
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.set_entry_point("a")
    g.set_finish_point("c")

    compiled = g.compile()
    result = await compiled.run({"text": "hi"})
    # "hi" → upper → "HI" → reverse → "IH" → prefix → "PREFIX_IH"
    assert result["text"] == "PREFIX_IH"


# ---------------------------------------------------------------------------
# 4. Conditional routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_conditional_routing_true_branch():
    async def decide(state: dict) -> str:
        return "long" if len(state.get("text", "")) > 3 else "short"

    async def add_long(state: dict) -> dict:
        return {"text": state["text"] + "_LONG"}

    async def add_short(state: dict) -> dict:
        return {"text": state["text"] + "_SHORT"}

    g = StateGraph()
    g.add_node("decide_node", lambda s: s)  # passthrough
    g.add_node("long_path", add_long)
    g.add_node("short_path", add_short)
    g.add_conditional_edge("decide_node", decide, {"long": "long_path", "short": "short_path"})
    g.set_finish_point("long_path")
    g.set_finish_point("short_path")
    g.set_entry_point("decide_node")
    g.allow_cycles = True

    compiled = g.compile(allow_cycles=False)
    result = await compiled.run({"text": "hello"})
    assert result["text"] == "hello_LONG"


@pytest.mark.asyncio
async def test_conditional_routing_false_branch():
    async def decide(state: dict) -> str:
        return "long" if len(state.get("text", "")) > 10 else "short"

    async def add_long(state: dict) -> dict:
        return {"text": state["text"] + "_LONG"}

    async def add_short(state: dict) -> dict:
        return {"text": state["text"] + "_SHORT"}

    g = StateGraph()
    g.add_node("start", lambda s: s)
    g.add_node("long_path", add_long)
    g.add_node("short_path", add_short)
    g.add_conditional_edge("start", decide, {"long": "long_path", "short": "short_path"})
    g.set_finish_point("long_path")
    g.set_finish_point("short_path")
    g.set_entry_point("start")

    compiled = g.compile(allow_cycles=False)
    result = await compiled.run({"text": "hi"})
    assert result["text"] == "hi_SHORT"


# ---------------------------------------------------------------------------
# 5. Stream yields node events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_stream_yields_events():
    g = StateGraph()
    g.add_node("upper", _upper)
    g.set_entry_point("upper")
    g.set_finish_point("upper")

    compiled = g.compile()
    events = []
    async for event in compiled.stream({"text": "abc"}):
        events.append(event)

    assert len(events) >= 1


# ---------------------------------------------------------------------------
# 6. run_sync works outside async context
# ---------------------------------------------------------------------------


def test_graph_run_sync():
    g = StateGraph()
    g.add_node("upper", _upper)
    g.set_entry_point("upper")
    g.set_finish_point("upper")

    compiled = g.compile()
    result = compiled.run_sync({"text": "sync"})
    assert result["text"] == "SYNC"


# ---------------------------------------------------------------------------
# 7. Validation errors
# ---------------------------------------------------------------------------


def test_graph_compile_without_entry_raises():
    g = StateGraph()
    g.add_node("a", _upper)
    g.set_finish_point("a")
    with pytest.raises(GraphConfigError, match="Entry point not set"):
        g.compile()


def test_graph_compile_nonexistent_entry_raises():
    g = StateGraph()
    g.add_node("a", _upper)
    g.set_entry_point("nonexistent")
    with pytest.raises(GraphConfigError, match="not a registered node"):
        g.compile()


def test_graph_cycle_detection_raises():
    g = StateGraph()

    async def noop(s):
        return s

    g.add_node("a", noop)
    g.add_node("b", noop)
    g.add_edge("a", "b")
    g.add_edge("b", "a")  # cycle
    g.set_entry_point("a")

    with pytest.raises(GraphConfigError, match="Cycle detected"):
        g.compile()


def test_graph_allow_cycles_bypasses_detection():
    g = StateGraph()

    async def noop(s):
        return s

    g.add_node("a", noop)
    g.add_node("b", noop)
    g.add_edge("a", "b")
    g.add_edge("b", "a")
    g.set_entry_point("a")

    # Should NOT raise with allow_cycles=True
    compiled = g.compile(allow_cycles=True, max_steps=5)
    assert compiled is not None


# ---------------------------------------------------------------------------
# 8. State passes through unchanged when node returns None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_node_returning_non_dict_raises():
    """Node that returns None (or non-dict) raises GraphRuntimeError."""
    from synapsekit.graph.errors import GraphRuntimeError

    async def bad_node(state: dict):
        return None  # violates the dict contract

    g = StateGraph()
    g.add_node("bad", bad_node)
    g.set_entry_point("bad")
    g.set_finish_point("bad")

    compiled = g.compile()
    with pytest.raises(GraphRuntimeError, match="must return a dict"):
        await compiled.run({"key": "value"})


# ---------------------------------------------------------------------------
# 9. StateGraph repr
# ---------------------------------------------------------------------------


def test_state_graph_repr():
    g = StateGraph()
    g.add_node("a", _upper)
    assert "StateGraph" in repr(g)
    assert "nodes=1" in repr(g)
