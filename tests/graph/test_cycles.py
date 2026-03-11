"""Tests for graph cycle support and configurable max_steps."""

import pytest

from synapsekit.graph.compiled import _MAX_STEPS
from synapsekit.graph.errors import GraphConfigError, GraphRuntimeError
from synapsekit.graph.graph import StateGraph
from synapsekit.graph.state import END

# ------------------------------------------------------------------ #
# allow_cycles=True skips cycle detection
# ------------------------------------------------------------------ #


async def test_allow_cycles_compiles_with_static_cycle():
    """Graph with a static cycle should compile when allow_cycles=True."""

    async def inc(state):
        return {"count": state.get("count", 0) + 1}

    g = StateGraph()
    g.add_node("a", inc).add_node("b", inc)
    g.add_edge("a", "b").add_edge("b", "a")
    g.set_entry_point("a")

    # Without allow_cycles, should raise
    with pytest.raises(GraphConfigError, match="Cycle detected"):
        g.compile()

    # With allow_cycles, should compile fine
    compiled = g.compile(allow_cycles=True, max_steps=5)
    assert compiled is not None


async def test_cycle_with_conditional_exit():
    """Cycle with a conditional exit should terminate."""

    async def inc(state):
        return {"count": state.get("count", 0) + 1}

    def route(state):
        return "done" if state["count"] >= 3 else "loop"

    g = StateGraph()
    g.add_node("inc", inc)
    g.add_conditional_edge("inc", route, {"loop": "inc", "done": END})
    g.set_entry_point("inc")

    result = await g.compile(allow_cycles=True).run({})
    assert result["count"] == 3


async def test_allow_cycles_false_detects_cycle():
    """Default behavior (allow_cycles=False) still detects static cycles."""

    async def noop(state):
        return {}

    g = StateGraph()
    g.add_node("a", noop).add_node("b", noop)
    g.add_edge("a", "b").add_edge("b", "a")
    g.set_entry_point("a")

    with pytest.raises(GraphConfigError, match="Cycle detected"):
        g.compile(allow_cycles=False)


# ------------------------------------------------------------------ #
# Configurable max_steps
# ------------------------------------------------------------------ #


async def test_custom_max_steps():
    """compile(max_steps=N) should limit execution to N steps."""

    async def inc(state):
        return {"count": state.get("count", 0) + 1}

    def always_loop(state):
        return "loop"

    g = StateGraph()
    g.add_node("loop", inc)
    g.add_conditional_edge("loop", always_loop, {"loop": "loop"})
    g.set_entry_point("loop")

    compiled = g.compile(allow_cycles=True, max_steps=5)
    with pytest.raises(GraphRuntimeError, match="_MAX_STEPS=5"):
        await compiled.run({})


async def test_default_max_steps_is_module_constant():
    """Default max_steps should match _MAX_STEPS = 100."""
    assert _MAX_STEPS == 100

    async def inc(state):
        return {"count": state.get("count", 0) + 1}

    g = StateGraph()
    g.add_node("n", inc)
    g.set_entry_point("n").set_finish_point("n")

    compiled = g.compile()
    assert compiled._max_steps == _MAX_STEPS


# ------------------------------------------------------------------ #
# Adjacency optimization
# ------------------------------------------------------------------ #


async def test_adjacency_index_built():
    """CompiledGraph should pre-build adjacency dict."""

    async def noop(state):
        return {}

    g = StateGraph()
    g.add_node("a", noop).add_node("b", noop)
    g.add_edge("a", "b")
    g.set_entry_point("a").set_finish_point("b")

    compiled = g.compile()
    assert "a" in compiled._adj
    assert len(compiled._adj["a"]) == 1  # edge a→b
    assert "b" in compiled._adj
    assert len(compiled._adj["b"]) == 1  # edge b→END
