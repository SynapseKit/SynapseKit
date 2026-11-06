"""Tests for SynapseKit Live — the zero-dependency event bus + SSE dashboard.

Real objects only (a real EventBus, a real stdlib HTTP server, real sockets via
urllib) — no mocks.
"""

from __future__ import annotations

import http.client
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


def test_publish_reserves_sequence_and_timestamp_fields() -> None:
    event_bus = EventBus()
    event_bus.enabled = True
    subscriber = event_bus.subscribe()
    before = time.time()

    event_bus.publish({"kind": "tool.call", "seq": 90, "ts": 0})

    after = time.time()
    history_event = event_bus.history()[0]
    delivered_event = subscriber.get(timeout=1)
    assert history_event["seq"] == 1
    assert delivered_event["seq"] == 1
    assert before <= history_event["ts"] <= after
    assert delivered_event["ts"] == history_event["ts"]


def test_publishers_do_not_wait_for_fanout_and_expose_sequence_gaps() -> None:
    event_bus = EventBus()
    event_bus.enabled = True
    subscriber = event_bus.subscribe()
    first_put = threading.Event()
    second_started = threading.Event()
    second_returned = threading.Event()
    release_first = threading.Event()
    put_nowait = subscriber.put_nowait

    def controlled_put(event: dict[str, object]) -> None:
        if event["seq"] == 1:
            first_put.set()
            release_first.wait(timeout=5)
        put_nowait(event)  # type: ignore[arg-type]

    def publish_second() -> None:
        second_started.set()
        event_bus.publish({"kind": "second"})
        second_returned.set()

    subscriber.put_nowait = controlled_put  # type: ignore[method-assign]
    first = threading.Thread(target=lambda: event_bus.publish({"kind": "first"}))
    second = threading.Thread(target=publish_second)
    first.start()
    try:
        assert first_put.wait(timeout=1)
        second.start()
        assert second_started.wait(timeout=1)
        second_returned_while_first_blocked = second_returned.wait(timeout=0.5)
    finally:
        release_first.set()
        first.join(timeout=2)
        if second.ident is not None:
            second.join(timeout=2)

    assert second_returned_while_first_blocked
    assert not first.is_alive() and not second.is_alive()

    event_bus.publish({"kind": "third"})
    delivered = [subscriber.get_nowait()["seq"] for _ in range(subscriber.qsize())]
    history = [event["seq"] for event in event_bus.history()]
    assert delivered == sorted(delivered)
    assert delivered == [1, 3]
    assert history == [1, 2, 3]
    assert set(history) - set(delivered) == {2}


def test_full_subscriber_queue_drops_events_with_a_visible_sequence_gap() -> None:
    event_bus = EventBus()
    event_bus.enabled = True
    subscriber = event_bus.subscribe()

    for index in range(subscriber.maxsize):
        event_bus.publish({"kind": "fill", "index": index})
    event_bus.publish({"kind": "dropped"})

    delivered = [subscriber.get_nowait()["seq"] for _ in range(subscriber.qsize())]
    event_bus.publish({"kind": "after_backpressure"})
    delivered.append(subscriber.get_nowait()["seq"])

    assert delivered == sorted(delivered)
    assert delivered == [*range(1, subscriber.maxsize + 1), subscriber.maxsize + 2]
    assert [event["seq"] for event in event_bus.history()][-2:] == [
        subscriber.maxsize + 1,
        subscriber.maxsize + 2,
    ]


def test_clear_preserves_subscriber_delivery_sequence() -> None:
    event_bus = EventBus()
    event_bus.enabled = True
    subscriber = event_bus.subscribe()

    event_bus.publish({"kind": "before_clear"})
    assert subscriber.get_nowait()["seq"] == 1
    event_bus.clear()
    event_bus.publish({"kind": "after_clear"})

    assert subscriber.get_nowait()["seq"] == 2


def test_clear_does_not_reuse_sequence_during_in_flight_delivery() -> None:
    event_bus = EventBus()
    event_bus.enabled = True
    subscriber = event_bus.subscribe()
    event_bus.publish({"kind": "queued_before_clear"})

    delivery_started = threading.Event()
    release_delivery = threading.Event()
    original_put = subscriber.put_nowait

    def controlled_put(event: dict[str, object]) -> None:
        if event["kind"] == "in_flight" and event["seq"] == 2:
            delivery_started.set()
            release_delivery.wait(timeout=5)
        original_put(event)  # type: ignore[arg-type]

    subscriber.put_nowait = controlled_put  # type: ignore[method-assign]
    publisher = threading.Thread(target=lambda: event_bus.publish({"kind": "in_flight"}))
    publisher.start()
    try:
        assert delivery_started.wait(timeout=1)
        event_bus.clear()
        assert event_bus.history() == []
    finally:
        release_delivery.set()
        publisher.join(timeout=2)

    assert not publisher.is_alive()
    event_bus.publish({"kind": "after_clear"})

    queued = [subscriber.get_nowait() for _ in range(3)]
    assert [event["seq"] for event in queued] == [1, 2, 3]
    assert [event["kind"] for event in queued] == [
        "queued_before_clear",
        "in_flight",
        "after_clear",
    ]
    assert [event["seq"] for event in event_bus.history()] == [3]


def test_clear_resets_sequence_when_no_subscribers_are_registered() -> None:
    event_bus = EventBus()
    event_bus.enabled = True
    event_bus.publish({"kind": "before_clear"})

    event_bus.clear()
    event_bus.publish({"kind": "after_clear"})

    assert [event["seq"] for event in event_bus.history()] == [1]


def test_history_replays_to_late_joiner() -> None:
    bus = EventBus()
    bus.enabled = True
    bus.publish({"kind": "a"})
    bus.publish({"kind": "b"})
    q = bus.subscribe()  # subscribes AFTER the events were published
    assert q.get(timeout=1)["kind"] == "a"
    assert q.get(timeout=1)["kind"] == "b"


def test_history_is_available_while_subscriber_fanout_is_blocked() -> None:
    event_bus = EventBus()
    event_bus.enabled = True
    subscriber = event_bus.subscribe()
    fanout_started = threading.Event()
    release_fanout = threading.Event()
    history_started = threading.Event()
    history_finished = threading.Event()
    original_put = subscriber.put_nowait

    def blocked_put(event: dict[str, object]) -> None:
        fanout_started.set()
        release_fanout.wait(timeout=5)
        original_put(event)  # type: ignore[arg-type]

    subscriber.put_nowait = blocked_put  # type: ignore[method-assign]
    publisher = threading.Thread(target=lambda: event_bus.publish({"kind": "blocked"}))
    history: list[dict[str, object]] = []

    def read_history() -> None:
        history_started.set()
        history.extend(event_bus.history())
        history_finished.set()

    reader = threading.Thread(target=read_history)
    publisher.start()
    try:
        assert fanout_started.wait(timeout=1)
        reader.start()
        assert history_started.wait(timeout=1)
        history_was_available_during_fanout = history_finished.wait(timeout=0.5)
    finally:
        release_fanout.set()
        publisher.join(timeout=2)
        if reader.ident is not None:
            reader.join(timeout=2)

    assert history_was_available_during_fanout
    assert not publisher.is_alive() and not reader.is_alive()
    assert [event["kind"] for event in history] == ["blocked"]


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


@pytest.mark.parametrize("path", ["/", "/index.html"])
def test_serves_dashboard_html_with_token(live_server, path: str) -> None:
    base, token = live_server
    html = urllib.request.urlopen(f"{base}{path}?token={token}", timeout=3).read().decode()
    assert "{{TOKEN}}" not in html  # placeholder replaced
    assert token in html  # real token injected
    assert "EventSource" in html and "SynapseKit" in html


@pytest.mark.parametrize(
    ("path", "query"),
    [
        ("/", ""),
        ("/", "?token=wrong"),
        ("/index.html", ""),
        ("/index.html", "?token=wrong"),
    ],
)
def test_dashboard_rejects_missing_or_wrong_token(live_server, path: str, query: str) -> None:
    base, _ = live_server
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(f"{base}{path}{query}", timeout=3)
    assert exc.value.code == 403


def test_dashboard_remains_public_when_token_is_disabled(live_server, monkeypatch) -> None:
    import synapsekit.live.server as live_server_module

    base, _ = live_server
    assert live_server_module._server is not None
    monkeypatch.setattr(live_server_module._server, "token", "")
    html = urllib.request.urlopen(base + "/", timeout=3).read().decode()
    assert "{{TOKEN}}" not in html


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
@pytest.mark.parametrize("path", ["/events", "/agui/events"])
def test_stop_revokes_active_sse_stream_after_restart(live_server, path: str) -> None:
    if path == "/agui/events":
        pytest.importorskip("ag_ui.core", reason="install synapsekit[ag-ui] for AG-UI tests")

    from synapsekit.live import bus

    base, old_token = live_server
    bus.clear()
    subscribers_before = bus.subscriber_count
    parsed = urllib.parse.urlparse(base)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=10)
    response: http.client.HTTPResponse | None = None
    reader: threading.Thread | None = None
    try:
        connection.request("GET", f"{path}?token={old_token}")
        response = connection.getresponse()
        assert response.status == 200
        assert response.readline() == b": connected\n"
        assert response.readline() == b"\n"
        assert bus.subscriber_count == subscribers_before + 1

        received: list[bytes] = []
        reader_ready = threading.Event()
        reader_done = threading.Event()

        def read_until_event_or_close() -> None:
            has_data = False
            reader_ready.set()
            try:
                while line := response.readline():
                    received.append(line)
                    has_data = has_data or line.startswith(b"data:")
                    if has_data and line == b"\n":
                        return
            except OSError:
                pass  # A reset is an expected way to revoke an SSE socket.
            finally:
                reader_done.set()

        reader = threading.Thread(target=read_until_event_or_close, daemon=True)
        reader.start()
        assert reader_ready.wait(timeout=1)

        stop()
        terminated_on_stop = reader_done.wait(timeout=1.5)
        subscribers_after_stop = bus.subscriber_count

        restarted_url = serve(port=0, quiet=True)
        restarted = urllib.parse.urlparse(restarted_url)
        new_token = urllib.parse.parse_qs(restarted.query)["token"][0]
        assert new_token != old_token
        if path == "/agui/events":
            bus.publish(
                {
                    "kind": "run.start",
                    "name": "run.start",
                    "status": "ok",
                    "attributes": {"label": "issue993-sentinel"},
                }
            )
        else:
            bus.publish({"kind": "issue993.sentinel", "marker": "issue993-sentinel"})
        reader_done.wait(timeout=2)

        assert terminated_on_stop, "the authenticated SSE client stayed open after stop()"
        assert subscribers_after_stop == subscribers_before, "stop() left the stream subscribed"
        assert not any(line.startswith(b"data:") for line in received), (
            "the stale connection received an event after the server restarted"
        )
    finally:
        if response is not None:
            response.close()
        connection.close()
        if reader is not None:
            reader.join(timeout=2)
        stop()


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
