"""Auto-instrumentation of the Ambient daemon → SynapseKit Live.

Real objects only — a real AmbientDaemon firing a real intervention against
injected source events. No server, no mocks: instrument the class and toggle
the bus directly.
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from pathlib import Path

import pytest

from synapsekit.ambient.daemon import AmbientDaemon, AmbientDaemonConfig
from synapsekit.ambient.events import AmbientEvent
from synapsekit.ambient.sources.base import AmbientSourcePlugin
from synapsekit.live import bus
from synapsekit.live.instrument import instrument_all


@pytest.fixture(autouse=True)
def _instrumented_bus():
    instrument_all()
    was = bus.enabled
    bus.enabled = True
    bus.clear()
    yield
    bus.enabled = was


class _FakeSource(AmbientSourcePlugin):
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


def test_fire_stays_a_coroutine_after_instrumentation() -> None:
    assert inspect.iscoroutinefunction(AmbientDaemon._fire)


async def test_ambient_intervention_publishes_event(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("synapsekit.ambient.daemon.notify_windows_toast", lambda *a: True)

    dirty = _event(
        "git", "git_status", "1 file", dirty=True, dirty_files=["foo.py"], branch="main"
    )
    # Raw command carries a secret; the Live event must not leak it.
    risky = _event("terminal", "command", "rm -rf build; export API_KEY=sk-secret-123")

    daemon = AmbientDaemon(
        config=AmbientDaemonConfig(
            status_path=tmp_path / "status.json", audit_path=tmp_path / "audit.jsonl"
        ),
        sources=[_FakeSource("git", [[dirty]]), _FakeSource("terminal", [[], [risky]])],
    )

    await daemon._tick()  # git dirty; terminal baseline
    await daemon._tick()  # risky command fires

    events = [e for e in bus.history() if e["kind"] == "ambient.intervene"]
    assert len(events) == 1
    attrs = events[0]["attributes"]
    assert attrs["rule"] == "destructive-delete"
    assert attrs["source"] == "terminal"
    assert 0.0 < attrs["confidence"] <= 1.0
    # No raw command text (with its secret) anywhere in the event.
    assert "sk-secret-123" not in str(events[0])
