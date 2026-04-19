"""Tests for subgraph checkpoint scoping.

Covers:
- Subgraph state is saved with a scoped graph_id, independent of parent
- Scoped ID follows parent::name::step format
- After a subgraph failure, resuming picks up from the failed step
- Without a checkpointer, subgraph behaviour is unchanged (no regression)
- Transient context keys never appear in final state or checkpoints
- Nested subgraphs get nested scopes
- Checkpointing composes with on_error strategies
- resume_subgraph convenience method works
"""

from __future__ import annotations

import pytest

from synapsekit.graph.checkpointers.memory import InMemoryCheckpointer
from synapsekit.graph.compiled import (
    _CHECKPOINTER_KEY,
    _GRAPH_ID_KEY,
    _STEP_KEY,
)
from synapsekit.graph.graph import StateGraph
from synapsekit.graph.subgraph import subgraph_node


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _make_subgraph(output: dict) -> object:
    """Return a compiled subgraph that always succeeds with *output*."""

    async def ok_node(state):
        return output

    g = StateGraph()
    g.add_node("ok", ok_node)
    g.set_entry_point("ok").set_finish_point("ok")
    return g.compile()


# ------------------------------------------------------------------ #
# Subgraph gets its own checkpoint
# ------------------------------------------------------------------ #


async def test_subgraph_gets_own_checkpoint():
    """Subgraph state is saved with a scoped graph_id via the parent checkpointer."""

    async def parent_node(state):
        return {"value": state.get("value", 0) + 1}

    async def sub_inner(state):
        return {"sub_result": state.get("value", 0) * 10}

    sub_graph = StateGraph()
    sub_graph.add_node("inner", sub_inner)
    sub_graph.set_entry_point("inner").set_finish_point("inner")

    parent = StateGraph()
    parent.add_node("step1", parent_node)
    parent.add_node("sub", subgraph_node(sub_graph.compile(), name="my_sub"))
    parent.add_edge("step1", "sub")
    parent.set_entry_point("step1").set_finish_point("sub")

    cp = InMemoryCheckpointer()
    result = await parent.compile().run(
        {"value": 5}, checkpointer=cp, graph_id="parent1"
    )

    assert result["value"] == 6
    assert result["sub_result"] == 60

    # The parent checkpoint should exist
    assert cp.load("parent1") is not None

    # The subgraph checkpoint should exist with scoped ID
    # Parent step for the "sub" node is step 2 (step1 runs at step 1, sub runs at step 2)
    sub_checkpoint = cp.load("parent1::my_sub::2")
    assert sub_checkpoint is not None


async def test_subgraph_checkpoint_id_format():
    """Verify the scoped ID format is parent_id::name::step."""

    async def noop(state):
        return {}

    sub = StateGraph()
    sub.add_node("n", noop)
    sub.set_entry_point("n").set_finish_point("n")

    parent = StateGraph()
    parent.add_node("sub", subgraph_node(sub.compile(), name="analyzer"))
    parent.set_entry_point("sub").set_finish_point("sub")

    cp = InMemoryCheckpointer()
    await parent.compile().run({}, checkpointer=cp, graph_id="workflow-42")

    # Subgraph runs at step 1 (it's the only / first node)
    assert cp.load("workflow-42::analyzer::1") is not None


# ------------------------------------------------------------------ #
# No checkpointer → no regression
# ------------------------------------------------------------------ #


async def test_no_checkpointer_no_regression():
    """Without a checkpointer, subgraph_node works exactly as before."""

    sub = _make_subgraph({"answer": 42})
    node_fn = subgraph_node(sub, name="test_sub")

    # No checkpoint context in state at all
    result = await node_fn({})
    assert result == {"answer": 42}


async def test_no_checkpointer_in_parent_graph_no_regression():
    """Running a parent graph without checkpointer should still work."""

    async def sub_inner(state):
        return {"result": "ok"}

    sub = StateGraph()
    sub.add_node("inner", sub_inner)
    sub.set_entry_point("inner").set_finish_point("inner")

    parent = StateGraph()
    parent.add_node("sub", subgraph_node(sub.compile(), name="s"))
    parent.set_entry_point("sub").set_finish_point("sub")

    result = await parent.compile().run({})
    assert result["result"] == "ok"


# ------------------------------------------------------------------ #
# Transient keys stripped
# ------------------------------------------------------------------ #


async def test_internal_keys_stripped_from_output():
    """Transient context keys must not appear in the final run() output."""

    async def echo(state):
        return {"echo": True}

    sub = StateGraph()
    sub.add_node("echo", echo)
    sub.set_entry_point("echo").set_finish_point("echo")

    parent = StateGraph()
    parent.add_node("sub", subgraph_node(sub.compile(), name="s"))
    parent.set_entry_point("sub").set_finish_point("sub")

    cp = InMemoryCheckpointer()
    result = await parent.compile().run({}, checkpointer=cp, graph_id="g1")

    assert _CHECKPOINTER_KEY not in result
    assert _GRAPH_ID_KEY not in result
    assert _STEP_KEY not in result


async def test_internal_keys_stripped_from_checkpoint():
    """Transient context keys must not be persisted in checkpoints."""

    async def inc(state):
        return {"count": state.get("count", 0) + 1}

    g = StateGraph()
    g.add_node("inc", inc)
    g.set_entry_point("inc").set_finish_point("inc")

    cp = InMemoryCheckpointer()
    await g.compile().run({"count": 0}, checkpointer=cp, graph_id="ck1")

    _, saved_state = cp.load("ck1")
    assert _CHECKPOINTER_KEY not in saved_state
    assert _GRAPH_ID_KEY not in saved_state
    assert _STEP_KEY not in saved_state


# ------------------------------------------------------------------ #
# Nested subgraphs get nested scopes
# ------------------------------------------------------------------ #


async def test_nested_subgraphs_get_nested_scopes():
    """A subgraph inside a subgraph gets a doubly-scoped checkpoint ID."""

    async def leaf(state):
        return {"leaf_done": True}

    inner_graph = StateGraph()
    inner_graph.add_node("leaf", leaf)
    inner_graph.set_entry_point("leaf").set_finish_point("leaf")

    mid_graph = StateGraph()
    mid_graph.add_node(
        "inner_sub", subgraph_node(inner_graph.compile(), name="inner")
    )
    mid_graph.set_entry_point("inner_sub").set_finish_point("inner_sub")

    outer_graph = StateGraph()
    outer_graph.add_node(
        "mid_sub", subgraph_node(mid_graph.compile(), name="middle")
    )
    outer_graph.set_entry_point("mid_sub").set_finish_point("mid_sub")

    cp = InMemoryCheckpointer()
    result = await outer_graph.compile().run(
        {}, checkpointer=cp, graph_id="root"
    )

    assert result["leaf_done"] is True

    # Middle subgraph checkpoint
    assert cp.load("root::middle::1") is not None

    # Inner subgraph checkpoint — nested scope
    assert cp.load("root::middle::1::inner::1") is not None


# ------------------------------------------------------------------ #
# Checkpointing composes with on_error strategies
# ------------------------------------------------------------------ #


async def test_subgraph_checkpoint_with_on_error_skip():
    """Checkpointing still works when on_error='skip' is used."""

    async def fail_node(state):
        raise RuntimeError("boom")

    sub = StateGraph()
    sub.add_node("fail", fail_node)
    sub.set_entry_point("fail").set_finish_point("fail")

    parent = StateGraph()
    parent.add_node(
        "sub", subgraph_node(sub.compile(), name="skipper", on_error="skip")
    )
    parent.set_entry_point("sub").set_finish_point("sub")

    cp = InMemoryCheckpointer()
    result = await parent.compile().run({}, checkpointer=cp, graph_id="skip1")

    # The error was caught — parent completed
    assert "__subgraph_error__" in result
    assert result["__subgraph_error__"]["type"] == "RuntimeError"

    # Parent checkpoint should exist
    assert cp.load("skip1") is not None


async def test_subgraph_checkpoint_with_on_error_fallback():
    """Fallback subgraph also gets checkpoint forwarding."""

    async def fail_node(state):
        raise RuntimeError("primary down")

    async def backup_node(state):
        return {"answer": 99}

    primary = StateGraph()
    primary.add_node("fail", fail_node)
    primary.set_entry_point("fail").set_finish_point("fail")

    backup = StateGraph()
    backup.add_node("backup", backup_node)
    backup.set_entry_point("backup").set_finish_point("backup")

    parent = StateGraph()
    parent.add_node(
        "sub",
        subgraph_node(
            primary.compile(),
            name="failable",
            on_error="fallback",
            fallback=backup.compile(),
        ),
    )
    parent.set_entry_point("sub").set_finish_point("sub")

    cp = InMemoryCheckpointer()
    result = await parent.compile().run({}, checkpointer=cp, graph_id="fb1")

    assert result["answer"] == 99
    assert "__subgraph_error__" in result


# ------------------------------------------------------------------ #
# resume_subgraph convenience method
# ------------------------------------------------------------------ #


async def test_resume_subgraph_method():
    """resume_subgraph() builds the correct scoped ID and delegates to resume()."""

    async def double(state):
        return {"value": state["value"] * 2}

    g = StateGraph()
    g.add_node("double", double)
    g.set_entry_point("double").set_finish_point("double")

    cp = InMemoryCheckpointer()

    # Manually save a checkpoint as if a subgraph had saved it
    cp.save("parent1::my_sub::2", 1, {"value": 5, "__synapsekit_graph_version": "1"})

    compiled = g.compile()
    result = await compiled.resume_subgraph(
        parent_graph_id="parent1",
        subgraph_name="my_sub",
        step=2,
        checkpointer=cp,
    )

    assert result["value"] == 10  # doubled from 5


# ------------------------------------------------------------------ #
# Default name
# ------------------------------------------------------------------ #


async def test_default_subgraph_name():
    """When no name is given, the default 'subgraph' is used in the scoped ID."""

    async def noop(state):
        return {}

    sub = StateGraph()
    sub.add_node("n", noop)
    sub.set_entry_point("n").set_finish_point("n")

    parent = StateGraph()
    parent.add_node("sub", subgraph_node(sub.compile()))  # no name= kwarg
    parent.set_entry_point("sub").set_finish_point("sub")

    cp = InMemoryCheckpointer()
    await parent.compile().run({}, checkpointer=cp, graph_id="p1")

    # Default name is "subgraph"
    assert cp.load("p1::subgraph::1") is not None
