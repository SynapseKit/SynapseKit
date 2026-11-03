"""Tests for agent.serve() — the FastAPI endpoint builder and the ServeMixin.

Uses a real FastAPI ``TestClient`` against ``build_agent_app`` (no running
server, no mocks) and hand-written fake agents, per the house testing rules.
``monkeypatch`` stands in for a missing optional dependency and records the
``uvicorn.run`` call without MagicMock.
"""

from __future__ import annotations

import json
import sys

import pytest
from fastapi.testclient import TestClient

from synapsekit import FunctionCallingAgent, ReActAgent, ReasoningAgent
from synapsekit.serve import ServeMixin, build_agent_app, serve_agent


class StreamingAgent:
    """Agent exposing both run() and stream() — serve() should pick streaming."""

    async def run(self, query: str) -> str:
        return f"ran: {query}"

    async def stream(self, query: str):
        for token in ("Hello", " ", "world"):
            yield token


class RunOnlyAgent:
    """Agent with only run() — serve() must fall back to a JSON response."""

    async def run(self, query: str) -> str:
        return f"ran: {query}"


class RunRaisesAgent:
    async def run(self, query: str) -> str:
        raise RuntimeError("secret internal detail")


class StreamRaisesAgent:
    async def stream(self, query: str):
        yield "first"
        raise RuntimeError("secret internal detail")


def test_streaming_agent_returns_sse_tokens_then_done() -> None:
    client = TestClient(build_agent_app(StreamingAgent()))

    with client.stream("POST", "/chat", json={"query": "hi"}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    tokens = [
        json.loads(line[len("data: ") :])
        for line in body.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert [t["token"] for t in tokens] == ["Hello", " ", "world"]
    assert body.strip().endswith("data: [DONE]")


class NonStringStreamAgent:
    """A custom agent that streams non-str chunks (ints) — must not abort."""

    async def stream(self, query: str):
        for value in (1, 2, 3):
            yield value


def test_non_string_stream_chunks_are_coerced_not_dropped() -> None:
    client = TestClient(build_agent_app(NonStringStreamAgent()))

    with client.stream("POST", "/chat", json={"query": "hi"}) as response:
        body = "".join(response.iter_text())

    tokens = [
        json.loads(line[len("data: ") :])["token"]
        for line in body.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert tokens == ["1", "2", "3"]
    assert body.strip().endswith("data: [DONE]")


def test_run_only_agent_returns_json_response() -> None:
    client = TestClient(build_agent_app(RunOnlyAgent()))

    response = client.post("/chat", json={"query": "hi"})

    assert response.status_code == 200
    assert response.json() == {"response": "ran: hi"}


def test_run_error_returns_generic_500_without_leaking_detail() -> None:
    client = TestClient(build_agent_app(RunRaisesAgent()), raise_server_exceptions=False)

    response = client.post("/chat", json={"query": "hi"})

    assert response.status_code == 500
    assert "secret internal detail" not in response.text
    assert response.json()["detail"] == "internal server error"


def test_stream_error_emits_generic_sse_error_without_leaking_detail() -> None:
    client = TestClient(build_agent_app(StreamRaisesAgent()))

    with client.stream("POST", "/chat", json={"query": "hi"}) as response:
        body = "".join(response.iter_text())

    assert "secret internal detail" not in body
    assert 'data: {"error": "internal server error"}' in body
    assert "data: [DONE]" not in body


def test_custom_path_is_respected() -> None:
    client = TestClient(build_agent_app(RunOnlyAgent(), path="/api/chat"))

    assert client.post("/api/chat", json={"query": "hi"}).status_code == 200
    assert client.post("/chat", json={"query": "hi"}).status_code == 404


def test_missing_query_field_is_a_422() -> None:
    client = TestClient(build_agent_app(RunOnlyAgent()))

    assert client.post("/chat", json={}).status_code == 422


@pytest.mark.parametrize("agent_cls", [ReActAgent, FunctionCallingAgent, ReasoningAgent])
def test_core_agents_inherit_serve_from_mixin(agent_cls: type) -> None:
    assert issubclass(agent_cls, ServeMixin)
    # The method is native (not monkeypatched on) and declared once on the mixin.
    assert agent_cls.serve is ServeMixin.serve


def test_serve_agent_invokes_uvicorn_with_host_and_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uvicorn

    calls: list[dict] = []
    monkeypatch.setattr(
        uvicorn, "run", lambda app, host, port: calls.append({"host": host, "port": port})
    )

    serve_agent(RunOnlyAgent(), host="127.0.0.1", port=9123, path="/chat")

    assert calls == [{"host": "127.0.0.1", "port": 9123}]


def test_mixin_serve_delegates_to_serve_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uvicorn

    calls: list[dict] = []
    monkeypatch.setattr(
        uvicorn, "run", lambda app, host, port: calls.append({"host": host, "port": port})
    )

    class MixedAgent(ServeMixin):
        async def run(self, query: str) -> str:
            return query

    MixedAgent().serve(host="0.0.0.0", port=8080)

    assert calls == [{"host": "0.0.0.0", "port": 8080}]


def test_serve_agent_raises_importerror_when_uvicorn_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "uvicorn", None)

    with pytest.raises(ImportError, match="synapsekit\\[serve\\]"):
        serve_agent(RunOnlyAgent())


def test_build_agent_app_raises_importerror_when_fastapi_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "fastapi", None)

    with pytest.raises(ImportError, match="synapsekit\\[serve\\]"):
        build_agent_app(RunOnlyAgent())
