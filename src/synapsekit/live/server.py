"""Zero-dependency localhost dashboard server for SynapseKit Live.

Uses only the standard library: ``http.server`` for HTTP and **Server-Sent
Events** (``text/event-stream``) for the live stream — no FastAPI, uvicorn, or
websockets. Serves one static ``dashboard.html`` plus a ``/events`` SSE endpoint
that fans the :data:`~synapsekit.live.bus.bus` out to the browser's built-in
``EventSource``.

Security: bound to ``127.0.0.1`` only, and ``/events`` requires a per-process
token (injected into the served HTML), so another local process can't read the
stream. Nothing ever leaves the machine.
"""

from __future__ import annotations

import json
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any
from urllib.parse import parse_qs, urlparse

from .bus import bus

# One running server per process.
_server: ThreadingHTTPServer | None = None
_url: str | None = None
_token: str = ""


def _dashboard_html() -> str:
    html = resources.files("synapsekit.live").joinpath("dashboard.html").read_text(encoding="utf-8")
    return html.replace("{{TOKEN}}", _token)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:  # keep the agent's stdout clean
        return None

    def _query(self) -> dict[str, list[str]]:
        return parse_qs(urlparse(self.path).query)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._serve_html()
        elif path == "/events":
            self._serve_events()
        elif path == "/healthz":
            self._serve_json({"ok": True, "subscribers": bus.subscriber_count})
        else:
            self.send_error(404, "Not found")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/approve":
            self.send_error(404, "Not found")
            return
        if _token and self._query().get("token", [""])[0] != _token:
            self.send_error(403, "Bad token")
            return
        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            data = {}
        from .approvals import resolve

        ok = resolve(data.get("id"), bool(data.get("approved")))
        self._serve_json({"ok": ok})

    def _serve_html(self) -> None:
        body = _dashboard_html().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_json(self, obj: dict[str, Any]) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_events(self) -> None:
        if _token and self._query().get("token", [""])[0] != _token:
            self.send_error(403, "Bad token")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        q = bus.subscribe()
        try:
            # Prime the connection so EventSource fires `open` immediately.
            self.wfile.write(b": connected\n\n")
            self.wfile.flush()
            while True:
                try:
                    event = q.get(timeout=15)
                except Exception:
                    # Heartbeat comment keeps the socket alive through proxies.
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    continue
                payload = json.dumps(event, default=str)
                self.wfile.write(f"data: {payload}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # browser tab closed
        finally:
            bus.unsubscribe(q)


def serve(
    host: str = "127.0.0.1",
    port: int = 7900,
    *,
    open_browser: bool = False,
    quiet: bool = False,
) -> str:
    """Start the Live dashboard server (idempotent). Returns the dashboard URL.

    Enables the event bus so subsequent :meth:`bus.publish` calls stream to any
    connected browser. Runs on a daemon thread, so it never blocks your program
    and dies with the process.
    """
    global _server, _url, _token
    if _server is not None and _url is not None:
        return _url

    _token = secrets.token_urlsafe(9)
    httpd = ThreadingHTTPServer((host, port), _Handler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, name="synapsekit-live", daemon=True)
    thread.start()

    _server = httpd
    bound_host = str(httpd.server_address[0])
    bound_port = int(httpd.server_address[1])
    _url = f"http://{bound_host}:{bound_port}/?token={_token}"
    bus.enabled = True

    if not quiet:
        print(f"◆ SynapseKit Live → {_url}")
    if open_browser:
        threading.Thread(target=lambda: webbrowser.open(_url or ""), daemon=True).start()
    return _url


def stop() -> None:
    """Stop the server and disable the bus (mainly for tests)."""
    global _server, _url
    if _server is not None:
        _server.shutdown()
        _server.server_close()
        _server = None
        _url = None
    bus.enabled = False


def current_url() -> str | None:
    return _url
