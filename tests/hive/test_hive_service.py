"""HTTP-boundary tests for the Hive FastAPI service and the urllib transport.

Two real boundaries, no mocks:

* ``create_hive_app`` is exercised through Starlette's ``TestClient`` (real ASGI
  round-trips) against a real ``HiveAggregator`` — routes, bearer auth, query
  params, and JSON (de)serialization.
* ``HttpHiveTransport`` uses stdlib ``urllib`` (not ``httpx``), so ``respx``
  can't intercept it; instead it is driven against a real localhost
  ``http.server`` bound to an ephemeral port, so the client's request building
  (method, path + ``_quote`` query string, bearer header, JSON body) and
  response parsing are tested over an actual TCP socket.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from synapsekit.hive import (
    ContributionEnvelope,
    ContributionPayload,
    HiveAggregator,
    HiveClient,
    HttpHiveTransport,
    PatternObservation,
    ShareScope,
    SQLiteHiveStore,
)

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient


def _client(tmp_path: Path, contributor: str) -> HiveClient:
    return HiveClient(
        scope=ShareScope.TEAM,
        team_id="synapsekit",
        contributor_id=contributor,
        cache_path=tmp_path / f"{contributor}.json",
    )


def _signed(client: HiveClient, key: str = "framework:fastapi") -> ContributionEnvelope:
    payload = ContributionPayload(
        scope=ShareScope.TEAM,
        scope_id=client.scope_id,
        patterns=(PatternObservation(key=key, value=1.0, category="framework"),),
        epsilon=1.0,
        delta=1e-6,
    )
    return client._sign(payload)


# --------------------------------------------------------------------------- #
# Server boundary: create_hive_app via TestClient
# --------------------------------------------------------------------------- #


def test_healthz_ok() -> None:
    app = create_app()
    with TestClient(app) as http:
        resp = http.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def create_app(**kwargs):
    from synapsekit.hive import create_hive_app

    return create_hive_app(HiveAggregator(SQLiteHiveStore(":memory:")), **kwargs)


def test_contribute_then_suggest_roundtrip(tmp_path: Path) -> None:
    aggregator = HiveAggregator(SQLiteHiveStore(":memory:"))
    from synapsekit.hive import create_hive_app

    app = create_hive_app(aggregator)
    with TestClient(app) as http:
        for name in ("alice", "bob", "carol"):
            resp = http.post("/v1/contributions", json=_signed(_client(tmp_path, name)).to_dict())
            assert resp.status_code == 200, resp.text
            assert resp.json()["contribution_id"]

        resp = http.get(
            "/v1/suggestions", params={"scope_id": "team:synapsekit", "minimum_cohort": 3}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["scope_id"] == "team:synapsekit"
    keys = {item["key"] for item in body["suggestions"]}
    assert "framework:fastapi" in keys


def test_bad_envelope_is_rejected_400() -> None:
    from synapsekit.hive import create_hive_app

    app = create_hive_app(HiveAggregator(SQLiteHiveStore(":memory:")))
    with TestClient(app) as http:
        resp = http.post("/v1/contributions", json={"not": "an envelope"})
    assert resp.status_code == 400


def test_bearer_auth_is_enforced() -> None:
    from synapsekit.hive import create_hive_app

    app = create_hive_app(HiveAggregator(SQLiteHiveStore(":memory:")), api_keys={"secret"})
    with TestClient(app) as http:
        missing = http.get("/v1/suggestions", params={"scope_id": "team:synapsekit"})
        bad = http.get(
            "/v1/suggestions",
            params={"scope_id": "team:synapsekit"},
            headers={"Authorization": "Bearer wrong"},
        )
        good = http.get(
            "/v1/suggestions",
            params={"scope_id": "team:synapsekit"},
            headers={"Authorization": "Bearer secret"},
        )
    assert missing.status_code == 401
    assert bad.status_code == 403
    assert good.status_code == 200


# --------------------------------------------------------------------------- #
# Client boundary: HttpHiveTransport (urllib) over a real socket
# --------------------------------------------------------------------------- #


class _RealHiveHandler(BaseHTTPRequestHandler):
    """A real HTTP endpoint backed by a real aggregator (set on the server)."""

    def log_message(self, *args: object) -> None:  # silence test noise
        pass

    def _send(self, status: int, payload: dict) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _record_auth(self) -> None:
        self.server.seen_auth.append(self.headers.get("Authorization"))  # type: ignore[attr-defined]

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length).decode("utf-8")) if length else {}

    def do_GET(self) -> None:
        self._record_auth()
        parts = urlsplit(self.path)
        params = {k: v[0] for k, v in parse_qs(parts.query).items()}
        agg: HiveAggregator = self.server.aggregator  # type: ignore[attr-defined]
        if parts.path == "/v1/suggestions":
            values = agg.suggestions(
                scope_id=params["scope_id"],
                query=params.get("query"),
                minimum_cohort=int(params.get("minimum_cohort", 3)),
                limit=int(params.get("limit", 20)),
            )
            self._send(
                200, {"scope_id": params["scope_id"], "suggestions": [v.to_dict() for v in values]}
            )
        elif parts.path == "/v1/transparency":
            report = agg.transparency(
                contributor_id=params["contributor_id"], scope_id=params["scope_id"]
            )
            self._send(200, report.to_dict())
        else:
            self._send(404, {"detail": "not found"})

    def do_POST(self) -> None:
        self._record_auth()
        parts = urlsplit(self.path)
        body = self._body()
        agg: HiveAggregator = self.server.aggregator  # type: ignore[attr-defined]
        if parts.path == "/v1/contributions":
            cid = agg.submit(ContributionEnvelope.from_dict(body))
            self._send(200, {"contribution_id": cid})
        elif parts.path == "/v1/withdraw":
            count = agg.withdraw(contributor_id=body["contributor_id"], scope_id=body["scope_id"])
            self._send(200, {"revoked": count})
        else:
            self._send(404, {"detail": "not found"})


@pytest.fixture
def live_hive_service():
    aggregator = HiveAggregator(SQLiteHiveStore(":memory:"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RealHiveHandler)
    server.aggregator = aggregator  # type: ignore[attr-defined]
    server.seen_auth = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield aggregator, f"http://{host}:{port}", server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


async def test_http_transport_full_roundtrip_over_real_socket(
    tmp_path: Path, live_hive_service
) -> None:
    _, base_url, server = live_hive_service
    transport = HttpHiveTransport(base_url, api_key="secret")

    # submit three real signed contributions through the urllib client
    for name in ("alice", "bob", "carol"):
        cid = await transport.submit(_signed(_client(tmp_path, name)))
        assert cid

    # GET with a _quote'd query string, JSON parsed back into Suggestions
    suggestions = await transport.suggestions(
        scope_id="team:synapsekit", query=None, minimum_cohort=3, limit=20
    )
    assert any(s.key == "framework:fastapi" for s in suggestions)

    # the bearer header actually crossed the socket
    assert "Bearer secret" in server.seen_auth


async def test_http_transport_withdraw_and_transparency(tmp_path: Path, live_hive_service) -> None:
    _, base_url, _ = live_hive_service
    client = _client(tmp_path, "alice")
    transport = HttpHiveTransport(base_url)
    await transport.submit(_signed(client))

    revoked = await transport.withdraw(
        contributor_id=client.contributor_id, scope_id="team:synapsekit"
    )
    assert revoked == 1

    report = await transport.transparency(
        contributor_id=client.contributor_id, scope_id="team:synapsekit"
    )
    assert report.scope_id == "team:synapsekit"
    assert report.withdrawn is True
