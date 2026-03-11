"""Tests for StateGraph builder and compile-time validation."""

import pytest

from synapsekit.graph.errors import GraphConfigError
from synapsekit.graph.graph import StateGraph
from synapsekit.graph.state import END


async def _noop(state):
    return {}


# ------------------------------------------------------------------ #
# Basic construction
# ------------------------------------------------------------------ #


def test_add_node_returns_self():
    g = StateGraph()
    result = g.add_node("a", _noop)
    assert result is g


def test_add_edge_returns_self():
    g = StateGraph()
    g.add_node("a", _noop).add_node("b", _noop)
    result = g.add_edge("a", "b")
    assert result is g


def test_set_entry_point_returns_self():
    g = StateGraph()
    g.add_node("a", _noop)
    result = g.set_entry_point("a")
    assert result is g


def test_set_finish_point_adds_end_edge():
    g = StateGraph()
    g.add_node("a", _noop).set_finish_point("a")
    assert any(getattr(e, "dst", None) == END for e in g._edges)


def test_compile_returns_compiled_graph():
    from synapsekit.graph.compiled import CompiledGraph

    g = StateGraph()
    g.add_node("a", _noop).set_entry_point("a")
    compiled = g.compile()
    assert isinstance(compiled, CompiledGraph)


# ------------------------------------------------------------------ #
# Validation errors
# ------------------------------------------------------------------ #


def test_compile_no_entry_point_raises():
    g = StateGraph()
    g.add_node("a", _noop)
    with pytest.raises(GraphConfigError, match="Entry point not set"):
        g.compile()


def test_compile_unknown_entry_point_raises():
    g = StateGraph()
    g.add_node("a", _noop)
    g.set_entry_point("nonexistent")
    with pytest.raises(GraphConfigError, match="not a registered node"):
        g.compile()


def test_compile_unknown_edge_src_raises():
    g = StateGraph()
    g.add_node("a", _noop)
    g.add_edge("ghost", "a")
    g.set_entry_point("a")
    with pytest.raises(GraphConfigError, match="not a registered node"):
        g.compile()


def test_compile_unknown_edge_dst_raises():
    g = StateGraph()
    g.add_node("a", _noop)
    g.add_edge("a", "ghost")
    g.set_entry_point("a")
    with pytest.raises(GraphConfigError, match="not a registered node"):
        g.compile()


def test_compile_unknown_conditional_mapping_dst_raises():
    g = StateGraph()
    g.add_node("a", _noop)
    g.add_conditional_edge("a", lambda s: "yes", {"yes": "ghost"})
    g.set_entry_point("a")
    with pytest.raises(GraphConfigError, match="not a registered node"):
        g.compile()


def test_edge_to_end_is_valid():
    g = StateGraph()
    g.add_node("a", _noop)
    g.add_edge("a", END)
    g.set_entry_point("a")
    g.compile()  # should not raise


def test_conditional_edge_mapping_to_end_is_valid():
    g = StateGraph()
    g.add_node("a", _noop)
    g.add_conditional_edge("a", lambda s: "stop", {"stop": END})
    g.set_entry_point("a")
    g.compile()  # should not raise


# ------------------------------------------------------------------ #
# Cycle detection
# ------------------------------------------------------------------ #


def test_cycle_detection_simple():
    g = StateGraph()
    g.add_node("a", _noop).add_node("b", _noop)
    g.add_edge("a", "b").add_edge("b", "a")
    g.set_entry_point("a")
    with pytest.raises(GraphConfigError, match="Cycle detected"):
        g.compile()


def test_cycle_detection_self_loop():
    g = StateGraph()
    g.add_node("a", _noop)
    g.add_edge("a", "a")
    g.set_entry_point("a")
    with pytest.raises(GraphConfigError, match="Cycle detected"):
        g.compile()


def test_no_false_positive_diamond():
    """a→b, a→c, b→d, c→d is a valid DAG (diamond), should not raise."""
    g = StateGraph()
    for name in ["a", "b", "c", "d"]:
        g.add_node(name, _noop)
    g.add_edge("a", "b").add_edge("a", "c")
    g.add_edge("b", "d").add_edge("c", "d")
    g.set_entry_point("a")
    g.compile()  # should not raise
