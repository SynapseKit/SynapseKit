"""Tests for graph checkpointing (InMemory and SQLite backends)."""

import pytest

from synapsekit.graph.checkpointers import (
    BaseCheckpointer,
    InMemoryCheckpointer,
    SQLiteCheckpointer,
)
from synapsekit.graph.errors import GraphRuntimeError
from synapsekit.graph.graph import StateGraph

# ------------------------------------------------------------------ #
# BaseCheckpointer ABC
# ------------------------------------------------------------------ #


def test_base_checkpointer_is_abstract():
    with pytest.raises(TypeError):
        BaseCheckpointer()  # type: ignore[abstract]


# ------------------------------------------------------------------ #
# InMemoryCheckpointer
# ------------------------------------------------------------------ #


class TestInMemoryCheckpointer:
    def test_save_and_load(self):
        cp = InMemoryCheckpointer()
        cp.save("g1", 3, {"a": 1})
        result = cp.load("g1")
        assert result is not None
        step, state = result
        assert step == 3
        assert state == {"a": 1}

    def test_load_missing_returns_none(self):
        cp = InMemoryCheckpointer()
        assert cp.load("nonexistent") is None

    def test_delete(self):
        cp = InMemoryCheckpointer()
        cp.save("g1", 1, {"x": 1})
        cp.delete("g1")
        assert cp.load("g1") is None

    def test_deepcopy_semantics(self):
        """Mutating saved state should not affect the checkpoint."""
        cp = InMemoryCheckpointer()
        state = {"items": [1, 2, 3]}
        cp.save("g1", 1, state)
        state["items"].append(4)  # mutate original

        _, loaded = cp.load("g1")
        assert loaded["items"] == [1, 2, 3]

    def test_overwrite(self):
        cp = InMemoryCheckpointer()
        cp.save("g1", 1, {"v": 1})
        cp.save("g1", 2, {"v": 2})
        step, state = cp.load("g1")
        assert step == 2
        assert state == {"v": 2}


# ------------------------------------------------------------------ #
# SQLiteCheckpointer
# ------------------------------------------------------------------ #


class TestSQLiteCheckpointer:
    def test_save_and_load(self):
        cp = SQLiteCheckpointer(":memory:")
        cp.save("g1", 3, {"a": 1})
        result = cp.load("g1")
        assert result is not None
        step, state = result
        assert step == 3
        assert state == {"a": 1}

    def test_load_missing_returns_none(self):
        cp = SQLiteCheckpointer(":memory:")
        assert cp.load("nonexistent") is None

    def test_delete(self):
        cp = SQLiteCheckpointer(":memory:")
        cp.save("g1", 1, {"x": 1})
        cp.delete("g1")
        assert cp.load("g1") is None

    def test_overwrite(self):
        cp = SQLiteCheckpointer(":memory:")
        cp.save("g1", 1, {"v": 1})
        cp.save("g1", 2, {"v": 2})
        step, state = cp.load("g1")
        assert step == 2
        assert state == {"v": 2}


# ------------------------------------------------------------------ #
# Integration: checkpointer wired into graph execution
# ------------------------------------------------------------------ #


async def test_run_with_checkpointer():
    """Graph run should save checkpoint after each wave."""

    async def add_greeting(state):
        return {"greeting": f"Hello, {state['name']}"}

    async def add_exclaim(state):
        return {"result": state["greeting"] + "!"}

    g = StateGraph()
    g.add_node("greet", add_greeting).add_node("exclaim", add_exclaim)
    g.add_edge("greet", "exclaim")
    g.set_entry_point("greet").set_finish_point("exclaim")

    cp = InMemoryCheckpointer()
    result = await g.compile().run({"name": "World"}, checkpointer=cp, graph_id="test1")

    assert result["result"] == "Hello, World!"
    # Should have saved a checkpoint
    loaded = cp.load("test1")
    assert loaded is not None
    step, state = loaded
    assert step >= 1
    assert state["result"] == "Hello, World!"


async def test_resume_from_checkpoint():
    """resume() should re-execute from checkpointed state."""

    async def double(state):
        return {"value": state["value"] * 2}

    g = StateGraph()
    g.add_node("double", double)
    g.set_entry_point("double").set_finish_point("double")

    cp = InMemoryCheckpointer()
    # First run
    await g.compile().run({"value": 5}, checkpointer=cp, graph_id="r1")

    # Resume — will re-run from saved state (value=10)
    compiled = g.compile()
    result = await compiled.resume("r1", cp)
    assert result["value"] == 20  # doubled again


async def test_resume_missing_checkpoint_raises():
    async def noop(state):
        return {}

    g = StateGraph()
    g.add_node("n", noop)
    g.set_entry_point("n").set_finish_point("n")

    cp = InMemoryCheckpointer()
    with pytest.raises(GraphRuntimeError, match="No checkpoint found"):
        await g.compile().resume("nonexistent", cp)


async def test_run_sync_with_checkpointer():
    """run_sync should forward checkpointer kwargs."""

    async def inc(state):
        return {"count": state.get("count", 0) + 1}

    g = StateGraph()
    g.add_node("inc", inc)
    g.set_entry_point("inc").set_finish_point("inc")

    cp = InMemoryCheckpointer()
    result = g.compile().run_sync({"count": 0}, checkpointer=cp, graph_id="sync1")
    assert result["count"] == 1
    assert cp.load("sync1") is not None


async def test_stream_with_checkpointer():
    """stream() should also save checkpoints."""

    async def inc(state):
        return {"count": state.get("count", 0) + 1}

    g = StateGraph()
    g.add_node("inc", inc)
    g.set_entry_point("inc").set_finish_point("inc")

    cp = InMemoryCheckpointer()
    compiled = g.compile()
    state = {"count": 0}
    events = []
    async for event in compiled.stream(state, checkpointer=cp, graph_id="s1"):
        events.append(event)

    assert len(events) == 1
    assert cp.load("s1") is not None


# ------------------------------------------------------------------ #
# Top-level exports
# ------------------------------------------------------------------ #


def test_top_level_exports():
    import synapsekit

    assert hasattr(synapsekit, "BaseCheckpointer")
    assert hasattr(synapsekit, "InMemoryCheckpointer")
    assert hasattr(synapsekit, "SQLiteCheckpointer")
