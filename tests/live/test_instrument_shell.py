"""Auto-instrumentation of the Agent OS Shell → SynapseKit Live.

Real objects only — a real ShellSession running a real (safe) subprocess. No
server, no mocks: instrument the class and toggle the bus directly.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from synapsekit.live import bus
from synapsekit.live.instrument import instrument_all
from synapsekit.shell import ShellHistory, ShellSession


@pytest.fixture(autouse=True)
def _instrumented_bus():
    instrument_all()
    was = bus.enabled
    bus.enabled = True
    bus.clear()
    yield
    bus.enabled = was


def _session(tmp_path: Path) -> ShellSession:
    return ShellSession(
        cwd=tmp_path,
        shell="bash",
        mesh=None,
        history=ShellHistory(tmp_path / "history.sqlite3"),
    )


def test_run_and_plan_stay_coroutines_after_instrumentation() -> None:
    assert inspect.iscoroutinefunction(ShellSession.run)
    assert inspect.iscoroutinefunction(ShellSession.plan)


async def test_shell_run_publishes_plan_and_run_events(tmp_path: Path) -> None:
    result = await _session(tmp_path).run("echo hello")
    assert result.ok

    kinds = [e["kind"] for e in bus.history()]
    assert "shell.plan" in kinds
    assert "shell.run" in kinds
    run_event = [e for e in bus.history() if e["kind"] == "shell.run"][-1]
    assert run_event["attributes"]["input"] == "echo hello"
    assert run_event["status"] == "ok"
    assert run_event["duration_ms"] >= 0
