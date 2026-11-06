"""Contract tests for the optional AG-UI Live adapter and its SSE route."""

from __future__ import annotations

import http.client
import json
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque

import pytest

from synapsekit.live import bus, serve
from synapsekit.live.server import stop


def _require_agui() -> tuple[object, dict[str, type]]:
    pytest.importorskip("ag_ui.core", reason="install synapsekit[ag-ui] for AG-UI tests")
    from ag_ui.core import (
        EventType,
        RunErrorEvent,
        RunStartedEvent,
        StateDeltaEvent,
        StateSnapshotEvent,
        TextMessageContentEvent,
        TextMessageEndEvent,
        TextMessageStartEvent,
        ToolCallEndEvent,
        ToolCallStartEvent,
    )

    return EventType, {
        "RunErrorEvent": RunErrorEvent,
        "RunStartedEvent": RunStartedEvent,
        "StateDeltaEvent": StateDeltaEvent,
        "StateSnapshotEvent": StateSnapshotEvent,
        "TextMessageContentEvent": TextMessageContentEvent,
        "TextMessageEndEvent": TextMessageEndEvent,
        "TextMessageStartEvent": TextMessageStartEvent,
        "ToolCallEndEvent": ToolCallEndEvent,
        "ToolCallStartEvent": ToolCallStartEvent,
    }


@pytest.fixture()
def live_server() -> tuple[str, str]:
    url = serve(port=0, quiet=True)
    parsed = urllib.parse.urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    token = urllib.parse.parse_qs(parsed.query)["token"][0]
    bus.clear()
    yield base, token
    stop()


def _open_stream(
    base: str, path: str
) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
    parsed = urllib.parse.urlparse(base)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=4)
    connection.request("GET", path)
    response = connection.getresponse()
    return connection, response


def _read_sse_data(response: http.client.HTTPResponse) -> dict[str, object]:
    data: list[str] = []
    while True:
        line = response.readline().decode("utf-8").rstrip("\r\n")
        if line.startswith("data:"):
            data.append(line[5:].lstrip())
        elif not line and data:
            return json.loads("\n".join(data))


def _abort_stream(
    connection: http.client.HTTPConnection,
    response: http.client.HTTPResponse,
    *,
    target_subscribers: int,
) -> None:
    response.close()
    connection.close()
    deadline = time.monotonic() + 3
    while bus.subscriber_count > target_subscribers and time.monotonic() < deadline:
        bus.publish({"kind": "run.start", "name": "run.start", "status": "ok", "attributes": {}})
        time.sleep(0.02)


def test_importing_live_does_not_import_optional_agui_package() -> None:
    script = "import sys; import synapsekit.live; assert 'ag_ui' not in sys.modules"
    subprocess.run([sys.executable, "-c", script], check=True, timeout=15)


def test_run_start_maps_to_run_started_without_inventing_finish() -> None:
    event_type, classes = _require_agui()
    from synapsekit.live.agui import AGUIEventAdapter

    adapter = AGUIEventAdapter()
    events = adapter.adapt(
        {
            "seq": 1,
            "kind": "run.start",
            "name": "run.start",
            "status": "ok",
            "attributes": {"label": "demo", "version": "2.0.1", "python": "3.14"},
        }
    )

    assert len(events) == 1
    assert isinstance(events[0], classes["RunStartedEvent"])
    assert events[0].type == event_type.RUN_STARTED
    assert events[0].thread_id and events[0].run_id
    assert not any(event.type == event_type.RUN_FINISHED for event in events)


def test_adapter_waits_for_run_start_before_mapping_events() -> None:
    event_type, _ = _require_agui()
    from synapsekit.live.agui import AGUIEventAdapter

    adapter = AGUIEventAdapter()
    before_start = adapter.adapt_many(
        [
            {
                "seq": 1,
                "kind": "tool.call",
                "name": "tool.call",
                "status": "ok",
                "attributes": {"tool": "pre-start"},
            },
            {
                "seq": 2,
                "kind": "graph.snapshot",
                "attributes": {"nodes": [{"id": "pre-start"}], "edges": []},
            },
        ]
    )
    assert before_start == []

    started = adapter.adapt(
        {"seq": 3, "kind": "run.start", "name": "run.start", "status": "ok", "attributes": {}}
    )
    after_start = adapter.adapt_many(
        [
            {
                "seq": 4,
                "kind": "tool.call",
                "name": "tool.call",
                "status": "ok",
                "attributes": {"tool": "after-start"},
            },
            {
                "seq": 5,
                "kind": "graph.snapshot",
                "attributes": {"nodes": [{"id": "after-start"}], "edges": []},
            },
        ]
    )

    assert [event.type for event in started] == [event_type.RUN_STARTED]
    assert [event.type for event in after_start] == [
        event_type.TOOL_CALL_START,
        event_type.TOOL_CALL_END,
        event_type.STATE_SNAPSHOT,
    ]


def test_sequence_gap_resets_run_context_until_the_next_run_start() -> None:
    event_type, _ = _require_agui()
    from synapsekit.live.agui import AGUIEventAdapter

    adapter = AGUIEventAdapter()
    started = adapter.adapt({"seq": 1, "kind": "run.start", "attributes": {}})
    before_gap = adapter.adapt(
        {"seq": 2, "kind": "tool.call", "attributes": {"tool": "before-gap"}}
    )
    after_missed_start = adapter.adapt(
        {"seq": 4, "kind": "tool.call", "attributes": {"tool": "after-missed-start"}}
    )
    while_waiting = adapter.adapt(
        {"seq": 5, "kind": "tool.call", "attributes": {"tool": "still-waiting"}}
    )
    restarted = adapter.adapt({"seq": 6, "kind": "run.start", "attributes": {}})
    after_restart = adapter.adapt(
        {"seq": 7, "kind": "tool.call", "attributes": {"tool": "after-restart"}}
    )

    assert [event.type for event in started] == [event_type.RUN_STARTED]
    assert [event.type for event in before_gap] == [
        event_type.TOOL_CALL_START,
        event_type.TOOL_CALL_END,
    ]
    assert after_missed_start == []
    assert while_waiting == []
    assert [event.type for event in restarted] == [event_type.RUN_STARTED]
    assert [event.type for event in after_restart] == [
        event_type.TOOL_CALL_START,
        event_type.TOOL_CALL_END,
    ]


def test_graph_snapshot_after_sequence_gap_is_a_full_snapshot() -> None:
    event_type, _ = _require_agui()
    from synapsekit.live.agui import AGUIEventAdapter

    adapter = AGUIEventAdapter()
    adapter.adapt({"seq": 1, "kind": "run.start", "attributes": {}})
    graph = {"nodes": [{"id": "first"}], "edges": []}
    first = adapter.adapt({"seq": 2, "kind": "graph.snapshot", "attributes": graph})
    delta = adapter.adapt({"seq": 3, "kind": "graph.snapshot", "attributes": graph})
    restarted = adapter.adapt({"seq": 5, "kind": "run.start", "attributes": {}})
    after_gap = adapter.adapt({"seq": 6, "kind": "graph.snapshot", "attributes": graph})

    assert [event.type for event in first] == [event_type.STATE_SNAPSHOT]
    assert [event.type for event in delta] == [event_type.STATE_DELTA]
    assert [event.type for event in restarted] == [event_type.RUN_STARTED]
    assert [event.type for event in after_gap] == [event_type.STATE_SNAPSHOT]


def test_graph_snapshot_starts_new_run_with_full_snapshot() -> None:
    event_type, _ = _require_agui()
    from synapsekit.live.agui import AGUIEventAdapter

    adapter = AGUIEventAdapter()
    graph = {"nodes": [{"id": "first"}], "edges": []}

    adapter.adapt({"seq": 1, "kind": "run.start", "attributes": {}})
    first = adapter.adapt({"seq": 2, "kind": "graph.snapshot", "attributes": graph})
    adapter.adapt({"seq": 3, "kind": "run.start", "attributes": {}})
    second = adapter.adapt({"seq": 4, "kind": "graph.snapshot", "attributes": graph})

    assert [event.type for event in first] == [event_type.STATE_SNAPSHOT]
    assert [event.type for event in second] == [event_type.STATE_SNAPSHOT]


def test_tool_call_maps_supported_name_without_fabricating_args_or_result() -> None:
    event_type, classes = _require_agui()
    from synapsekit.live.agui import AGUIEventAdapter

    adapter = AGUIEventAdapter()
    adapter.adapt({"kind": "run.start", "name": "run.start", "attributes": {}})
    events = adapter.adapt(
        {
            "seq": 2,
            "kind": "tool.call",
            "name": "tool.call",
            "status": "ok",
            "duration_ms": 1.5,
            "attributes": {"tool": "notion", "operation": "search"},
        }
    )

    assert [event.type for event in events] == [
        event_type.TOOL_CALL_START,
        event_type.TOOL_CALL_END,
    ]
    assert isinstance(events[0], classes["ToolCallStartEvent"])
    assert isinstance(events[1], classes["ToolCallEndEvent"])
    assert events[0].tool_call_name == "notion"
    assert not any(
        event.type in {event_type.TOOL_CALL_ARGS, event_type.TOOL_CALL_RESULT} for event in events
    )


def test_llm_span_emits_text_only_from_a_real_output_string() -> None:
    event_type, classes = _require_agui()
    from synapsekit.live.agui import AGUIEventAdapter

    adapter = AGUIEventAdapter()
    adapter.adapt({"kind": "run.start", "name": "run.start", "attributes": {}})
    events = adapter.adapt(
        {
            "seq": 3,
            "kind": "span",
            "name": "llm.generate",
            "status": "ok",
            "attributes": {"llm.input": "private prompt", "llm.output": "answer from model"},
        }
    )

    assert [event.type for event in events] == [
        event_type.TEXT_MESSAGE_START,
        event_type.TEXT_MESSAGE_CONTENT,
        event_type.TEXT_MESSAGE_END,
    ]
    assert isinstance(events[0], classes["TextMessageStartEvent"])
    assert isinstance(events[1], classes["TextMessageContentEvent"])
    assert isinstance(events[2], classes["TextMessageEndEvent"])
    assert events[1].delta == "answer from model"
    assert "private prompt" not in repr(events)
    assert (
        adapter.adapt(
            {
                "kind": "span",
                "name": "llm.generate",
                "status": "ok",
                "attributes": {"llm.input": "private prompt"},
            }
        )
        == []
    )


def test_error_mapping_does_not_expose_traceback() -> None:
    event_type, classes = _require_agui()
    from synapsekit.live.agui import AGUIEventAdapter

    adapter = AGUIEventAdapter()
    error_source = {
        "seq": 1,
        "kind": "span",
        "name": "agent.step",
        "status": "error",
        "attributes": {"traceback": "SECRET traceback and local paths"},
    }
    assert adapter.adapt(error_source) == []
    adapter.adapt({"seq": 2, "kind": "run.start", "name": "run.start", "attributes": {}})
    events = adapter.adapt({**error_source, "seq": 3})

    assert len(events) == 1
    assert isinstance(events[0], classes["RunErrorEvent"])
    assert events[0].type == event_type.RUN_ERROR
    assert "SECRET" not in repr(events[0])
    assert "local paths" not in repr(events[0])


def test_graph_snapshots_then_emit_valid_replace_patch() -> None:
    event_type, classes = _require_agui()
    from synapsekit.live.agui import AGUIEventAdapter

    adapter = AGUIEventAdapter()
    adapter.adapt({"kind": "run.start", "name": "run.start", "attributes": {}})
    first_graph = {"nodes": [{"id": "a"}], "edges": []}
    second_graph = {"nodes": [{"id": "a"}, {"id": "b"}], "edges": [["a", "b"]]}
    first = adapter.adapt({"seq": 5, "kind": "graph.snapshot", "attributes": first_graph})
    second = adapter.adapt({"seq": 6, "kind": "graph.snapshot", "attributes": second_graph})

    assert len(first) == len(second) == 1
    assert isinstance(first[0], classes["StateSnapshotEvent"])
    assert first[0].type == event_type.STATE_SNAPSHOT
    assert first[0].snapshot == {"graph": first_graph}
    assert isinstance(second[0], classes["StateDeltaEvent"])
    assert second[0].type == event_type.STATE_DELTA
    patch = second[0].delta[0]
    assert patch.op == "replace"
    assert patch.path == "/graph"
    assert patch.value == second_graph


def test_unsupported_events_are_ignored() -> None:
    _require_agui()
    from synapsekit.live.agui import AGUIEventAdapter

    assert AGUIEventAdapter().adapt({"seq": 7, "kind": "trace", "attributes": {"tree": {}}}) == []
    assert AGUIEventAdapter().adapt({"seq": 8, "kind": "llm.call", "model": "claude"}) == []


def test_adapter_emits_events_in_source_sequence_order() -> None:
    event_type, _ = _require_agui()
    from synapsekit.live.agui import AGUIEventAdapter

    events = AGUIEventAdapter().adapt_many(
        [
            {"seq": 3, "kind": "tool.call", "status": "ok", "attributes": {"tool": "later"}},
            {"seq": 1, "kind": "run.start", "attributes": {}},
            {"seq": 2, "kind": "graph.snapshot", "attributes": {"nodes": [], "edges": []}},
        ]
    )

    assert [event.type for event in events] == [
        event_type.RUN_STARTED,
        event_type.STATE_SNAPSHOT,
        event_type.TOOL_CALL_START,
        event_type.TOOL_CALL_END,
    ]


def test_agui_route_rejects_bad_token_before_subscribing(live_server: tuple[str, str]) -> None:
    base, _ = live_server
    before = bus.subscriber_count

    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(f"{base}/agui/events?token=wrong", timeout=3)

    assert error.value.code == 403
    assert bus.subscriber_count == before


def test_agui_route_returns_clear_503_if_optional_sdk_is_missing(
    live_server: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    base, token = live_server
    monkeypatch.delitem(sys.modules, "ag_ui.encoder", raising=False)
    monkeypatch.setitem(sys.modules, "ag_ui", None)

    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(f"{base}/agui/events?token={token}", timeout=3)

    assert error.value.code == 503
    assert "synapsekit[ag-ui]" in error.value.read().decode("utf-8")


def test_authorized_agui_route_uses_sdk_encoded_sse(live_server: tuple[str, str]) -> None:
    _require_agui()
    base, token = live_server
    connection, response = _open_stream(base, f"/agui/events?token={token}")
    try:
        assert response.status == 200
        assert response.getheader("Content-Type").startswith("text/event-stream")
        assert response.readline().decode().strip() == ": connected"
        assert response.readline().decode().strip() == ""
        bus.publish({"kind": "run.start", "name": "run.start", "status": "ok", "attributes": {}})
        event = _read_sse_data(response)
    finally:
        _abort_stream(connection, response, target_subscribers=0)

    assert event["type"] == "RUN_STARTED"
    assert event.get("runId") or event.get("run_id")
    assert event.get("threadId") or event.get("thread_id")


def test_agui_late_subscriber_waits_for_next_start_if_history_evicts_start(
    live_server: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_agui()
    base, token = live_server
    monkeypatch.setattr(bus, "_history", deque(maxlen=1))
    bus.publish({"kind": "run.start", "name": "run.start", "status": "ok", "attributes": {}})
    bus.publish(
        {
            "kind": "tool.call",
            "name": "tool.call",
            "status": "ok",
            "attributes": {"tool": "before-start"},
        }
    )
    assert [event["kind"] for event in bus.history()] == ["tool.call"]
    connection, response = _open_stream(base, f"/agui/events?token={token}")
    try:
        assert response.status == 200
        assert response.readline().decode().strip() == ": connected"
        assert response.readline().decode().strip() == ""
        bus.publish({"kind": "run.start", "name": "run.start", "status": "ok", "attributes": {}})
        bus.publish(
            {
                "kind": "tool.call",
                "name": "tool.call",
                "status": "ok",
                "attributes": {"tool": "after-start"},
            }
        )
        received = [_read_sse_data(response), _read_sse_data(response), _read_sse_data(response)]
    finally:
        _abort_stream(connection, response, target_subscribers=0)

    # The evicted start leaves only a tool call in replay; it is ignored until
    # the next actual run.start, after which the new tool call is mapped.
    assert [event["type"] for event in received] == [
        "RUN_STARTED",
        "TOOL_CALL_START",
        "TOOL_CALL_END",
    ]


def test_agui_disconnect_cleans_subscriber_without_finishing_run(
    live_server: tuple[str, str],
) -> None:
    _require_agui()
    base, token = live_server
    before = bus.subscriber_count
    connection, response = _open_stream(base, f"/agui/events?token={token}")
    assert response.status == 200
    assert response.readline().decode().strip() == ": connected"
    assert response.readline().decode().strip() == ""
    assert bus.subscriber_count == before + 1

    _abort_stream(connection, response, target_subscribers=before)

    assert bus.subscriber_count == before
    assert all(event["kind"] != "run.end" for event in bus.history())


def test_dashboard_events_route_keeps_raw_live_event_json(live_server: tuple[str, str]) -> None:
    base, token = live_server
    connection, response = _open_stream(base, f"/events?token={token}")
    try:
        assert response.status == 200
        assert response.readline().decode().strip() == ": connected"
        assert response.readline().decode().strip() == ""
        bus.publish({"kind": "llm.call", "model": "dashboard-regression"})
        data = _read_sse_data(response)
    finally:
        _abort_stream(connection, response, target_subscribers=0)

    assert data["kind"] == "llm.call"
    assert data["model"] == "dashboard-regression"
    assert "type" not in data
