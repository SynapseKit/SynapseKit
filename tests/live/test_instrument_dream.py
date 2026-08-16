"""Dream Mode → SynapseKit Live.

Real objects only — a real DreamMode running against a real local SQLite state
store and audit dir. No server, no mocks: toggle the bus directly and assert
the published run event. Dream publishes directly (not via the auto-wrapper)
because the meaningful telemetry is the run *result*.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from synapsekit.dream import DreamConfig, DreamMode, PowerStatus
from synapsekit.live import bus


@pytest.fixture(autouse=True)
def _live_bus():
    was = bus.enabled
    bus.enabled = True
    bus.clear()
    yield
    bus.enabled = was


def _mode(tmp_path: Path) -> DreamMode:
    return DreamMode(
        config=DreamConfig(
            schedule="02:00",
            state_path=tmp_path / "state.sqlite3",
            audit_dir=tmp_path / "audit",
        )
    )


async def test_completed_run_publishes_dream_run(tmp_path: Path) -> None:
    mode = _mode(tmp_path)
    try:
        await mode.run_once(force=True, power=PowerStatus(plugged_in=True, battery_percent=100))
    finally:
        mode.close()

    events = [e for e in bus.history() if e["kind"] == "dream.run"]
    assert len(events) == 1
    assert events[0]["status"] == "ok"
    assert events[0]["attributes"]["run_status"] == "completed"
    assert "traces_replayed" in events[0]["attributes"]


async def test_skipped_run_publishes_dream_run_with_reason(tmp_path: Path) -> None:
    mode = _mode(tmp_path)
    try:
        # Unplugged: run_once skips (force bypasses the schedule, not the power gate).
        await mode.run_once(force=True, power=PowerStatus(plugged_in=False, battery_percent=50))
    finally:
        mode.close()

    events = [e for e in bus.history() if e["kind"] == "dream.run"]
    assert len(events) == 1
    assert events[0]["attributes"]["run_status"] == "skipped"
    assert "plugged-in" in events[0]["attributes"]["skipped_reason"]


async def test_no_event_published_when_live_disabled(tmp_path: Path) -> None:
    bus.enabled = False
    mode = _mode(tmp_path)
    try:
        await mode.run_once(force=True, power=PowerStatus(plugged_in=True, battery_percent=100))
    finally:
        mode.close()
    assert [e for e in bus.history() if e["kind"] == "dream.run"] == []
