"""Tests for subgraph_node on_error strategies: raise, retry, fallback, skip."""

from __future__ import annotations

import pytest

from synapsekit.graph.graph import StateGraph
from synapsekit.graph.subgraph import subgraph_node

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _make_ok_subgraph(output: dict) -> object:
    """Return a compiled subgraph that always succeeds with *output*."""

    async def ok_node(state):
        return output

    g = StateGraph()
    g.add_node("ok", ok_node)
    g.set_entry_point("ok").set_finish_point("ok")
    return g.compile()


def _make_fail_subgraph(exc: Exception) -> object:
    """Return a compiled subgraph whose single node always raises *exc*."""

    async def fail_node(state):
        raise exc

    g = StateGraph()
    g.add_node("fail", fail_node)
    g.set_entry_point("fail").set_finish_point("fail")
    return g.compile()


def _make_flaky_subgraph(fail_n: int, output: dict) -> object:
    """Subgraph that fails on the first *fail_n* calls then succeeds."""
    call_count = {"n": 0}

    async def flaky_node(state):
        call_count["n"] += 1
        if call_count["n"] <= fail_n:
            raise RuntimeError(f"transient failure #{call_count['n']}")
        return output

    g = StateGraph()
    g.add_node("flaky", flaky_node)
    g.set_entry_point("flaky").set_finish_point("flaky")
    return g.compile()


# ------------------------------------------------------------------ #
# on_error="raise" (default)
# ------------------------------------------------------------------ #


async def test_raise_propagates_exception():
    sub = _make_fail_subgraph(ValueError("boom"))
    node_fn = subgraph_node(sub)  # default on_error="raise"

    with pytest.raises(ValueError, match="boom"):
        await node_fn({})


async def test_raise_succeeds_when_no_error():
    sub = _make_ok_subgraph({"result": 42})
    node_fn = subgraph_node(sub)

    out = await node_fn({})
    assert out == {"result": 42}


# ------------------------------------------------------------------ #
# on_error="retry"
# ------------------------------------------------------------------ #


async def test_retry_succeeds_after_failures():
    sub = _make_flaky_subgraph(fail_n=2, output={"value": "ok"})
    node_fn = subgraph_node(sub, on_error="retry", max_retries=3)

    out = await node_fn({})
    assert out == {"value": "ok"}


async def test_retry_raises_after_all_attempts_exhausted():
    sub = _make_flaky_subgraph(fail_n=10, output={})
    node_fn = subgraph_node(sub, on_error="retry", max_retries=3)

    with pytest.raises(Exception, match="3 attempt"):
        await node_fn({})


async def test_retry_default_max_retries_is_3():
    """Default max_retries=3 → fail 4 times → should raise."""
    sub = _make_flaky_subgraph(fail_n=10, output={})
    node_fn = subgraph_node(sub, on_error="retry")

    with pytest.raises(Exception, match="3 attempt"):
        await node_fn({})


async def test_retry_succeeds_on_first_try():
    sub = _make_ok_subgraph({"x": 1})
    node_fn = subgraph_node(sub, on_error="retry", max_retries=5)

    out = await node_fn({})
    assert out == {"x": 1}


# ------------------------------------------------------------------ #
# on_error="fallback"
# ------------------------------------------------------------------ #


async def test_fallback_runs_when_primary_fails():
    primary = _make_fail_subgraph(RuntimeError("primary down"))
    backup = _make_ok_subgraph({"result": "from_fallback"})
    node_fn = subgraph_node(primary, on_error="fallback", fallback=backup)

    out = await node_fn({})
    assert out["result"] == "from_fallback"


async def test_fallback_sets_error_info_in_output():
    primary = _make_fail_subgraph(RuntimeError("primary down"))
    backup = _make_ok_subgraph({"result": "from_fallback"})
    node_fn = subgraph_node(primary, on_error="fallback", fallback=backup)

    out = await node_fn({})
    assert "__subgraph_error__" in out
    err = out["__subgraph_error__"]
    assert err["type"] == "RuntimeError"
    assert "primary down" in err["message"]
    assert err["attempts"] == 1


async def test_fallback_not_called_when_primary_succeeds():
    primary = _make_ok_subgraph({"result": "primary_ok"})
    # fallback would fail if called
    backup = _make_fail_subgraph(AssertionError("should not be called"))
    node_fn = subgraph_node(primary, on_error="fallback", fallback=backup)

    out = await node_fn({})
    assert out["result"] == "primary_ok"
    assert "__subgraph_error__" not in out


async def test_fallback_requires_fallback_graph():
    with pytest.raises(ValueError, match="fallback CompiledGraph"):
        subgraph_node(_make_ok_subgraph({}), on_error="fallback")


# ------------------------------------------------------------------ #
# on_error="skip"
# ------------------------------------------------------------------ #


async def test_skip_returns_empty_on_failure():
    sub = _make_fail_subgraph(KeyError("missing"))
    node_fn = subgraph_node(sub, on_error="skip")

    out = await node_fn({})
    # No primary output keys, but error info is present
    assert "__subgraph_error__" in out


async def test_skip_contains_error_info():
    sub = _make_fail_subgraph(TypeError("type mismatch"))
    node_fn = subgraph_node(sub, on_error="skip")

    out = await node_fn({})
    err = out["__subgraph_error__"]
    assert err["type"] == "TypeError"
    assert "type mismatch" in err["message"]


async def test_skip_succeeds_transparently_when_no_error():
    sub = _make_ok_subgraph({"data": "hello"})
    node_fn = subgraph_node(sub, on_error="skip")

    out = await node_fn({})
    assert out == {"data": "hello"}
    assert "__subgraph_error__" not in out


# ------------------------------------------------------------------ #
# Integration — subgraph inside a parent graph
# ------------------------------------------------------------------ #


async def test_skip_in_parent_graph_continues_execution():
    """Parent graph should keep running after a skipped failing subgraph."""
    failing_sub = _make_fail_subgraph(RuntimeError("sub failed"))
    sub_node = subgraph_node(failing_sub, on_error="skip")

    async def after_sub(state):
        err = state.get("__subgraph_error__")
        return {"continued": True, "had_error": err is not None}

    parent = StateGraph()
    parent.add_node("sub", sub_node)
    parent.add_node("after", after_sub)
    parent.add_edge("sub", "after")
    parent.set_entry_point("sub").set_finish_point("after")

    result = await parent.compile().run({})
    assert result["continued"] is True
    assert result["had_error"] is True


async def test_fallback_in_parent_graph():
    """Fallback subgraph output is visible to downstream nodes."""
    primary = _make_fail_subgraph(RuntimeError("down"))
    backup = _make_ok_subgraph({"answer": 99})

    sub_node = subgraph_node(primary, on_error="fallback", fallback=backup)

    async def consumer(state):
        return {"final": state["answer"] + 1}

    parent = StateGraph()
    parent.add_node("sub", sub_node)
    parent.add_node("consumer", consumer)
    parent.add_edge("sub", "consumer")
    parent.set_entry_point("sub").set_finish_point("consumer")

    result = await parent.compile().run({})
    assert result["final"] == 100


# ------------------------------------------------------------------ #
# Input / output mapping still works with error strategies
# ------------------------------------------------------------------ #


async def test_output_mapping_applied_on_success_with_retry():
    sub = _make_ok_subgraph({"out": "hello"})
    node_fn = subgraph_node(
        sub,
        output_mapping={"out": "parent_out"},
        on_error="retry",
        max_retries=2,
    )

    out = await node_fn({})
    assert out == {"parent_out": "hello"}


# ------------------------------------------------------------------ #
# Validation
# ------------------------------------------------------------------ #


def test_invalid_max_retries_raises():
    sub = _make_ok_subgraph({})
    with pytest.raises(ValueError, match="max_retries"):
        subgraph_node(sub, on_error="retry", max_retries=0)
