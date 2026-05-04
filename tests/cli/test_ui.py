"""Tests for the observability UI server endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from synapsekit.cli.ui_server import create_app


@pytest.fixture()
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


# ---------------------------------------------------------------------------
# /api/health
# ---------------------------------------------------------------------------


def test_health_endpoint_returns_ok(client: TestClient) -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data == {"status": "ok"}


def test_health_endpoint_status_key_is_string(client: TestClient) -> None:
    resp = client.get("/api/health")
    assert isinstance(resp.json()["status"], str)


# ---------------------------------------------------------------------------
# /api/metrics
# ---------------------------------------------------------------------------


def test_metrics_endpoint_returns_200(client: TestClient) -> None:
    resp = client.get("/api/metrics")
    assert resp.status_code == 200


def test_metrics_endpoint_has_required_keys(client: TestClient) -> None:
    data = client.get("/api/metrics").json()
    required = {
        "total_calls",
        "total_tokens",
        "total_cost_usd",
        "avg_latency_ms",
    }
    for key in required:
        assert key in data, f"Missing key: {key!r}"


def test_metrics_total_calls_is_int(client: TestClient) -> None:
    data = client.get("/api/metrics").json()
    assert isinstance(data["total_calls"], int)


def test_metrics_total_cost_usd_is_float_or_int(client: TestClient) -> None:
    data = client.get("/api/metrics").json()
    assert isinstance(data["total_cost_usd"], (float, int))


def test_metrics_avg_latency_ms_is_numeric(client: TestClient) -> None:
    data = client.get("/api/metrics").json()
    assert isinstance(data["avg_latency_ms"], (float, int))


def test_metrics_empty_tracer_returns_zero_calls(client: TestClient) -> None:
    data = client.get("/api/metrics").json()
    assert data["total_calls"] == 0


# ---------------------------------------------------------------------------
# /api/traces
# ---------------------------------------------------------------------------


def test_traces_endpoint_returns_list(client: TestClient) -> None:
    resp = client.get("/api/traces")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_traces_empty_by_default(client: TestClient) -> None:
    data = client.get("/api/traces").json()
    assert data == []


def test_traces_after_recording(client: TestClient) -> None:
    # Seed a record into the tracer
    tracer = client.app.state.tracer  # type: ignore[attr-defined]
    tracer.record(input_tokens=10, output_tokens=5, latency_ms=123.4)
    data = client.get("/api/traces").json()
    assert len(data) >= 1
    trace = data[0]
    assert "input_tokens" in trace
    assert "output_tokens" in trace
    assert "latency_ms" in trace


# ---------------------------------------------------------------------------
# Dashboard HTML
# ---------------------------------------------------------------------------


def test_root_returns_html(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "SynapseKit" in resp.text


# ---------------------------------------------------------------------------
# create_app accepts optional tracer
# ---------------------------------------------------------------------------


def test_create_app_with_custom_tracer() -> None:
    from synapsekit.observability.tracer import TokenTracer

    tracer = TokenTracer(model="gpt-4o")
    tracer.record(input_tokens=100, output_tokens=50, latency_ms=200.0)
    app = create_app(tracer=tracer)
    c = TestClient(app)
    data = c.get("/api/metrics").json()
    assert data["total_calls"] == 1
    assert data["total_tokens"] == 150
