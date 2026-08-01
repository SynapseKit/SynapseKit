"""Tests for SynapseKit Live — the zero-dependency event bus + SSE dashboard.

Real objects only (a real EventBus, a real stdlib HTTP server, real sockets via
urllib) — no mocks.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import pytest

from synapsekit.live import serve
from synapsekit.live.bus import EventBus
from synapsekit.live.server import stop


# ── EventBus (pure, no server) ───────────────────────────────────────────────
def test_publish_is_noop_when_disabled() -> None:
    bus = EventBus()
    assert bus.enabled is False
    bus.publish({"kind": "x"})
    assert bus.history() == []  # nothing recorded, no work done


def test_publish_fans_out_to_subscribers() -> None:
    bus = EventBus()
    bus.enabled = True
    q = bus.subscribe()
    bus.publish({"kind": "tool.call", "tool": "notion"})
    event = q.get(timeout=1)
    assert event["kind"] == "tool.call"
    assert event["tool"] == "notion"
    assert event["seq"] == 1 and "ts" in event


def test_history_replays_to_late_joiner() -> None:
    bus = EventBus()
    bus.enabled = True
    bus.publish({"kind": "a"})
    bus.publish({"kind": "b"})
    q = bus.subscribe()  # subscribes AFTER the events were published
    assert q.get(timeout=1)["kind"] == "a"
    assert q.get(timeout=1)["kind"] == "b"


def test_history_is_bounded() -> None:
    bus = EventBus(history=5)
    bus.enabled = True
    for i in range(20):
        bus.publish({"kind": "e", "i": i})
    hist = bus.history()
    assert len(hist) == 5
    assert [e["i"] for e in hist] == [15, 16, 17, 18, 19]


def test_unsubscribe_stops_delivery() -> None:
    bus = EventBus()
    bus.enabled = True
    q = bus.subscribe()
    bus.unsubscribe(q)
    bus.publish({"kind": "z"})
    assert bus.subscriber_count == 0


# ── SSE server (real HTTP) ────────────────────────────────────────────────────
@pytest.fixture()
def live_server():
    url = serve(port=0, quiet=True)  # OS-assigned free port
    base = url.split("/?")[0]
    token = urllib.parse.urlparse(url).query.split("token=")[1]
    yield base, token
    stop()


def test_serves_dashboard_html_with_token(live_server) -> None:
    base, token = live_server
    html = urllib.request.urlopen(base + "/", timeout=3).read().decode()
    assert "{{TOKEN}}" not in html  # placeholder replaced
    assert token in html  # real token injected
    assert "EventSource" in html and "SynapseKit" in html


def test_healthz(live_server) -> None:
    base, _ = live_server
    body = json.loads(urllib.request.urlopen(base + "/healthz", timeout=3).read())
    assert body["ok"] is True


def test_events_stream_delivers_published_events(live_server) -> None:
    from synapsekit.live import bus

    base, token = live_server
    bus.clear()  # drop any history from earlier tests so replay is deterministic
    received: list[str] = []

    def reader() -> None:
        req = urllib.request.Request(f"{base}/events?token={token}")
        with urllib.request.urlopen(req, timeout=5) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if line.startswith("data:"):
                    received.append(line[5:].strip())
                    if len(received) >= 2:
                        return

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    time.sleep(0.3)  # let the reader connect
    bus.publish({"kind": "llm.call", "model": "claude-opus-4-8"})
    bus.publish({"kind": "span", "name": "retriever.search"})
    t.join(timeout=4)

    kinds = [json.loads(r)["kind"] for r in received]
    assert "llm.call" in kinds


def test_events_rejects_bad_token(live_server) -> None:
    base, _ = live_server
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(f"{base}/events?token=wrong", timeout=3)
    assert exc.value.code == 403


# ── observe → bus integration ────────────────────────────────────────────────
def test_observe_span_publishes_to_bus(live_server) -> None:
    from synapsekit.live import bus
    from synapsekit.observe.runtime import InMemoryExporter, configure, end_span, start_span

    configure(exporter=InMemoryExporter())  # enable instrumentation (silent)
    before = len(bus.history())
    span = start_span("llm.call", attributes={"model": "claude-opus-4-8"})
    end_span(span)
    after = bus.history()
    assert len(after) > before
    assert any(e.get("name") == "llm.call" for e in after)
