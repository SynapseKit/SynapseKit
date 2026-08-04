"""Tests for AmbientDaemon polling, notification, and lifecycle."""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from synapsekit.ambient.daemon import AmbientDaemon, AmbientDaemonConfig
from synapsekit.ambient.events import AmbientEvent
from synapsekit.ambient.sources.base import AmbientSourcePlugin


class _FakeSource(AmbientSourcePlugin):
    """Injectable source plugin that replays a fixed sequence of poll() batches."""

    def __init__(self, name: str, batches: list[list[AmbientEvent]]) -> None:
        self.name = name
        self._batches = [list(batch) for batch in batches]

    async def poll(self) -> list[AmbientEvent]:
        if self._batches:
            return self._batches.pop(0)
        return []


def _event(source: str, kind: str, text: str, **metadata: object) -> AmbientEvent:
    return AmbientEvent(source=source, kind=kind, text=text, timestamp=datetime.now(UTC), metadata=metadata)


@pytest.mark.asyncio
async def test_tick_fires_notification_and_records_audit(tmp_path: Path, monkeypatch) -> None:
    notified = []
    monkeypatch.setattr(
        "synapsekit.ambient.daemon.notify_windows_toast",
        lambda title, message: notified.append((title, message)) or True,
    )

    dirty_event = _event(
        "git", "git_status", "1 file changed", dirty=True, dirty_files=["foo.py"], branch="main"
    )
    risky_event = _event("terminal", "command", "rm -rf build")

    git_source = _FakeSource("git", [[dirty_event]])
    terminal_source = _FakeSource("terminal", [[], [risky_event]])

    config = AmbientDaemonConfig(
        status_path=tmp_path / "status.json",
        audit_path=tmp_path / "audit.jsonl",
    )
    daemon = AmbientDaemon(config=config, sources=[git_source, terminal_source])

    await daemon._tick()  # git reports dirty; terminal baseline (empty)
    assert notified == []

    await daemon._tick()  # terminal reports the risky command against dirty state
    assert len(notified) == 1
    assert "rm -rf build" in notified[0][1]

    entries = daemon.audit_log.query()
    assert len(entries) == 1
    assert entries[0].model == "destructive-delete"
    assert entries[0].input_text == "rm -rf build"


@pytest.mark.asyncio
async def test_low_confidence_intervention_is_suppressed(tmp_path: Path, monkeypatch) -> None:
    notified = []
    monkeypatch.setattr(
        "synapsekit.ambient.daemon.notify_windows_toast",
        lambda title, message: notified.append((title, message)) or True,
    )

    dirty_event = _event("git", "git_status", "1 file", dirty=True, dirty_files=["foo.py"], branch="main")
    risky_event = _event("terminal", "command", "git clean -fd")

    git_source = _FakeSource("git", [[dirty_event]])
    terminal_source = _FakeSource("terminal", [[], [risky_event]])

    config = AmbientDaemonConfig(
        status_path=tmp_path / "status.json",
        audit_path=tmp_path / "audit.jsonl",
        min_confidence=0.95,  # above every rule's confidence
    )
    daemon = AmbientDaemon(config=config, sources=[git_source, terminal_source])

    await daemon._tick()
    await daemon._tick()
    assert notified == []


@pytest.mark.asyncio
async def test_start_records_pid_and_stop_marks_stopped(tmp_path: Path) -> None:
    config = AmbientDaemonConfig(
        status_path=tmp_path / "status.json",
        audit_path=tmp_path / "audit.jsonl",
        poll_interval=0.05,
    )
    daemon = AmbientDaemon(config=config, sources=[])

    task = asyncio.create_task(daemon.start())
    try:
        await asyncio.sleep(0.1)
        status = daemon.status()
        assert status.state == "running"
        assert status.pid == os.getpid()
    finally:
        await daemon.stop()
        await asyncio.wait_for(task, timeout=5)

    assert daemon.status().state == "stopped"
