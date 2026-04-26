"""Tests for TimeBasedAutoResume."""

from __future__ import annotations

import inspect
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from synapsekit.graph import InMemoryCheckpointer, TimeBasedAutoResume
from synapsekit.graph.time_resume import schedule_resume

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_compiled_graph(*, raises: Exception | None = None, result: dict | None = None):
    """Return a minimal mock CompiledGraph."""
    mock = MagicMock()
    if raises is not None:
        mock.run = AsyncMock(side_effect=raises)
        mock.resume = AsyncMock(side_effect=raises)
    else:
        mock.run = AsyncMock(return_value=result or {"output": "ok"})
        mock.resume = AsyncMock(return_value=result or {"output": "ok"})
    return mock


# ---------------------------------------------------------------------------
# Constructor & schedule_resume method
# ---------------------------------------------------------------------------


def test_time_based_auto_resume_constructor():
    cg = _make_compiled_graph()
    cp = InMemoryCheckpointer()
    tar = TimeBasedAutoResume(
        compiled_graph=cg,
        checkpointer=cp,
        resume_after_seconds=30.0,
        max_retries=2,
        poll_interval=5.0,
    )
    assert tar.resume_after_seconds == 30.0
    assert tar.max_retries == 2
    assert tar.poll_interval == 5.0
    assert tar._poll_task is None


def test_schedule_resume_returns_time_based_auto_resume():
    cg = _make_compiled_graph()
    cp = InMemoryCheckpointer()
    # Attach the convenience helper
    import asyncio

    async def _run():
        result = schedule_resume(cg, "t1", cp, after_seconds=60.0)
        assert isinstance(result, TimeBasedAutoResume)
        assert result.resume_after_seconds == 60.0
        # Cancel to clean up task
        if result._poll_task:
            result._poll_task.cancel()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# start() — success path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_success():
    cg = _make_compiled_graph(result={"answer": 42})
    cp = InMemoryCheckpointer()
    tar = TimeBasedAutoResume(compiled_graph=cg, checkpointer=cp, poll_interval=9999.0)

    result = await tar.start("t1", {"q": "hi"})
    assert result == {"answer": 42}

    st = await tar.status("t1")
    assert st["completed"] is True
    assert st["retries"] == 0
    assert st["next_resume_at"] is None

    # Cleanup background task
    if tar._poll_task:
        tar._poll_task.cancel()


# ---------------------------------------------------------------------------
# start() — failure path, interrupted_at is recorded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_failure_records_interrupted_at():
    cg = _make_compiled_graph(raises=RuntimeError("boom"))
    cp = InMemoryCheckpointer()
    tar = TimeBasedAutoResume(
        compiled_graph=cg, checkpointer=cp, resume_after_seconds=100.0, poll_interval=9999.0
    )

    with pytest.raises(RuntimeError, match="boom"):
        await tar.start("t1", {"q": "hi"})

    st = await tar.status("t1")
    assert st["completed"] is False
    assert st["retries"] == 0
    assert st["next_resume_at"] is not None
    assert st["next_resume_at"] > time.time()  # in the future

    if tar._poll_task:
        tar._poll_task.cancel()


# ---------------------------------------------------------------------------
# cancel()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_prevents_resume():
    cg = _make_compiled_graph(raises=RuntimeError("fail"))
    cp = InMemoryCheckpointer()
    tar = TimeBasedAutoResume(compiled_graph=cg, checkpointer=cp, poll_interval=9999.0)

    with pytest.raises(RuntimeError):
        await tar.start("t1", {})

    await tar.cancel("t1")
    st = await tar.status("t1")
    assert st["cancelled"] is True

    if tar._poll_task:
        tar._poll_task.cancel()


# ---------------------------------------------------------------------------
# status() — unknown thread
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_unknown_thread():
    tar = TimeBasedAutoResume(
        compiled_graph=_make_compiled_graph(),
        checkpointer=InMemoryCheckpointer(),
    )
    st = await tar.status("nonexistent")
    assert st["thread_id"] == "nonexistent"
    assert st["state"] is None
    assert st["retries"] == 0
    assert st["next_resume_at"] is None
    assert st["completed"] is False


# ---------------------------------------------------------------------------
# _poll_loop() — triggers resume when time has elapsed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_poll_loop_triggers_resume():
    import asyncio

    cg = _make_compiled_graph(result={"done": True})
    cp = InMemoryCheckpointer()
    tar = TimeBasedAutoResume(
        compiled_graph=cg,
        checkpointer=cp,
        resume_after_seconds=0.01,  # essentially immediate
        max_retries=1,
        poll_interval=0.05,
    )

    # Manually seed an interrupted thread
    tar._threads["t2"] = {
        "state": {"x": 1},
        "retries": 0,
        "interrupted_at": time.time() - 1.0,  # already past resume_after_seconds
        "cancelled": False,
        "result": None,
        "error": None,
    }

    tar._ensure_poller()
    # Give the poller a chance to fire
    await asyncio.sleep(0.2)

    assert cg.resume.called
    assert tar._threads["t2"]["result"] == {"done": True}

    if tar._poll_task:
        tar._poll_task.cancel()


# ---------------------------------------------------------------------------
# max_retries not exceeded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_poll_loop_respects_max_retries():
    import asyncio

    cg = _make_compiled_graph(raises=RuntimeError("still failing"))
    cp = InMemoryCheckpointer()
    tar = TimeBasedAutoResume(
        compiled_graph=cg,
        checkpointer=cp,
        resume_after_seconds=0.01,
        max_retries=2,
        poll_interval=0.05,
    )

    tar._threads["t3"] = {
        "state": {},
        "retries": 0,
        "interrupted_at": time.time() - 1.0,
        "cancelled": False,
        "result": None,
        "error": None,
    }

    tar._ensure_poller()
    await asyncio.sleep(0.5)

    # Should have retried up to max_retries
    assert tar._threads["t3"]["retries"] <= 2

    if tar._poll_task:
        tar._poll_task.cancel()


# ---------------------------------------------------------------------------
# TimeBasedAutoResume is exported from graph.__init__
# ---------------------------------------------------------------------------


def test_exported_from_graph_init():
    from synapsekit import graph

    assert hasattr(graph, "TimeBasedAutoResume")
    assert graph.TimeBasedAutoResume is TimeBasedAutoResume


# ---------------------------------------------------------------------------
# Decorator / coroutine check
# ---------------------------------------------------------------------------


def test_start_is_coroutine():
    tar = TimeBasedAutoResume(
        compiled_graph=_make_compiled_graph(),
        checkpointer=InMemoryCheckpointer(),
    )
    assert inspect.iscoroutinefunction(tar.start)
    assert inspect.iscoroutinefunction(tar.cancel)
    assert inspect.iscoroutinefunction(tar.status)
