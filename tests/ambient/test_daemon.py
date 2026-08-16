"""Tests for AmbientDaemon polling, notification, and lifecycle."""

from __future__ import annotations

import asyncio
import inspect
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from synapsekit.ambient.daemon import AmbientDaemon, AmbientDaemonConfig
from synapsekit.ambient.events import AmbientEvent
from synapsekit.ambient.sources.base import AmbientSourcePlugin
from synapsekit.ambient.status import write_status


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
    return AmbientEvent(
        source=source, kind=kind, text=text, timestamp=datetime.now(UTC), metadata=metadata
    )


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
async def test_secret_in_command_is_redacted_in_audit(tmp_path: Path, monkeypatch) -> None:
    # A risky command may also carry a secret; it must not be persisted
    # verbatim to the on-disk audit log.
    monkeypatch.setattr(
        "synapsekit.ambient.daemon.notify_windows_toast", lambda *args: True
    )

    dirty_event = _event(
        "git", "git_status", "1 file", dirty=True, dirty_files=["foo.py"], branch="main"
    )
    risky_event = _event("terminal", "command", "rm -rf build; export API_KEY=sk-secret-123")

    git_source = _FakeSource("git", [[dirty_event]])
    terminal_source = _FakeSource("terminal", [[], [risky_event]])

    config = AmbientDaemonConfig(
        status_path=tmp_path / "status.json",
        audit_path=tmp_path / "audit.jsonl",
    )
    daemon = AmbientDaemon(config=config, sources=[git_source, terminal_source])

    await daemon._tick()
    await daemon._tick()

    entries = daemon.audit_log.query()
    assert len(entries) == 1
    assert "sk-secret-123" not in entries[0].input_text
    assert "REDACTED" in entries[0].input_text


@pytest.mark.asyncio
async def test_low_confidence_intervention_is_suppressed(tmp_path: Path, monkeypatch) -> None:
    notified = []
    monkeypatch.setattr(
        "synapsekit.ambient.daemon.notify_windows_toast",
        lambda title, message: notified.append((title, message)) or True,
    )

    dirty_event = _event(
        "git", "git_status", "1 file", dirty=True, dirty_files=["foo.py"], branch="main"
    )
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


@pytest.mark.asyncio
async def test_stop_keeps_status_running_when_target_process_survives(
    tmp_path: Path, monkeypatch
) -> None:
    # If the signaled process is still alive after the timeout, stop() must
    # NOT lie by marking it stopped with pid=None (that orphans a live
    # daemon and discards the pid needed to signal it again).
    monkeypatch.setattr("synapsekit.ambient.daemon._STOP_TIMEOUT_SECONDS", 0.15)

    class _NoKillDaemon(AmbientDaemon):
        # Pretend we signaled the pid, but never actually kill it, so the
        # target "survives" the stop request.
        @staticmethod
        def _signal_pid(pid: int) -> bool:
            return True

    status_path = tmp_path / "status.json"
    live_pid = os.getppid()  # a real, alive process that is not this test
    write_status(status_path, state="running", pid=live_pid)

    config = AmbientDaemonConfig(status_path=status_path, audit_path=tmp_path / "a.jsonl")
    daemon = _NoKillDaemon(config=config, sources=[])

    result = await daemon.stop()

    assert result.state == "running"
    assert result.pid == live_pid


def test_status_reports_stopped_when_recorded_pid_is_dead(tmp_path: Path) -> None:
    # A daemon killed with SIGKILL leaves a stale "running" status file;
    # status() must reconcile it against process liveness.
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    dead_pid = proc.pid  # reaped: no longer a live process

    status_path = tmp_path / "status.json"
    write_status(status_path, state="running", pid=dead_pid)

    daemon = AmbientDaemon(config=AmbientDaemonConfig(status_path=status_path), sources=[])

    assert daemon.status().state == "stopped"


def test_fire_is_async_so_blocking_io_stays_off_the_loop() -> None:
    # The toast + jsonl audit write block; _fire must remain a coroutine
    # that offloads them, or the poll loop stalls (async-first guarantee).
    assert inspect.iscoroutinefunction(AmbientDaemon._fire)
    assert inspect.iscoroutinefunction(AmbientDaemon._tick)
