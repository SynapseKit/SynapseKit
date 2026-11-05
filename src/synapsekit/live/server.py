"""Zero-dependency localhost dashboard server for SynapseKit Live.

Uses only the standard library: ``http.server`` for HTTP and **Server-Sent
Events** (``text/event-stream``) for the live stream — no FastAPI, uvicorn, or
websockets. Serves one static ``dashboard.html`` plus a ``/events`` SSE endpoint
that fans the :data:`~synapsekit.live.bus.bus` out to the browser's built-in
``EventSource``.

Security: defaults to ``127.0.0.1``; callers may select another bind host.
The dashboard and event streams require a per-process token, which is injected
into authorized dashboard HTML. Use non-loopback binds only on trusted networks.
"""

from __future__ import annotations

import contextlib
import json
import queue
import secrets
import socket
import threading
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

from .bus import bus

# One running server per process.
_server: _LiveHTTPServer | None = None
_url: str | None = None
_token: str = ""
_lifecycle_lock = threading.Lock()


@dataclass(eq=False)
class _StreamSession:
    """One authenticated SSE client, revocable independently of the listener."""

    connection: socket.socket
    events: queue.Queue[dict[str, Any]]
    stopped: threading.Event = field(default_factory=threading.Event)

    def revoke(self) -> None:
        self.stopped.set()
        bus.unsubscribe(self.events)
        # Wake a handler blocked in Queue.get(); a full queue is already awake.
        with contextlib.suppress(queue.Full):
            self.events.put_nowait({})
        # Also interrupt writes blocked on a client that stopped reading.
        with contextlib.suppress(OSError):
            self.connection.shutdown(socket.SHUT_RDWR)


class _LiveHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], token: str) -> None:
        self.token = token
        self._streams_lock = threading.Lock()
        self._streams: set[_StreamSession] = set()
        self._stopping = False
        super().__init__(server_address, _Handler)

    def open_stream(self, connection: socket.socket) -> _StreamSession | None:
        with self._streams_lock:
            if self._stopping:
                return None
            session = _StreamSession(connection, bus.subscribe())
            self._streams.add(session)
            return session

    def close_stream(self, session: _StreamSession) -> None:
        with self._streams_lock:
            self._streams.discard(session)
        bus.unsubscribe(session.events)

    def revoke_streams(self) -> None:
        with self._streams_lock:
            self._stopping = True
            sessions = tuple(self._streams)
        for session in sessions:
            session.revoke()


def _dashboard_html(token: str | None = None) -> str:

    token = _token if token is None else token
    html = resources.files("synapsekit.live").joinpath("dashboard.html").read_text(encoding="utf-8")
    return html.replace("{{TOKEN}}", token)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:  # keep the agent's stdout clean
        return None

    @property
    def _live_server(self) -> _LiveHTTPServer:
        return cast(_LiveHTTPServer, self.server)

    def _authorized(self) -> bool:
        token = self._live_server.token
        return not token or self._query().get("token", [""])[0] == token

    def _query(self) -> dict[str, list[str]]:
        return parse_qs(urlparse(self.path).query)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            if not self._authorized():
                self.send_error(403, "Bad token")
                return
            self._serve_html()
        elif path == "/events":
            self._serve_events()
        elif path == "/agui/events":
            self._serve_agui_events()
        elif path == "/healthz":
            self._serve_json({"ok": True, "subscribers": bus.subscriber_count})
        else:
            self.send_error(404, "Not found")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/approve":
            self.send_error(404, "Not found")
            return
        if not self._authorized():
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
        body = _dashboard_html(self._live_server.token).encode("utf-8")
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
        if not self._authorized():
            self.send_error(403, "Bad token")
            return
        session = self._live_server.open_stream(self.connection)
        if session is None:
            self.send_error(503, "Server is stopping")
            return
        try:
            self._run_stream(session)
        except (BrokenPipeError, ConnectionResetError, OSError, ValueError):
            pass  # Client disconnect or concurrent server shutdown.
        finally:
            self._live_server.close_stream(session)

    def _serve_agui_events(self) -> None:
        if not self._authorized():
            self.send_error(403, "Bad token")
            return

        from .agui import AGUIDependencyError, AGUIEventAdapter, create_event_encoder

        try:
            adapter = AGUIEventAdapter()
            encoder = create_event_encoder(self.headers.get("accept"))
        except AGUIDependencyError as exc:
            self.send_error(503, str(exc))
            return

        session = self._live_server.open_stream(self.connection)
        if session is None:
            self.send_error(503, "Server is stopping")
            return
        try:
            self._run_stream(session, adapter=adapter, encoder=encoder)
        except (BrokenPipeError, ConnectionResetError, OSError, ValueError):
            pass
        finally:
            self._live_server.close_stream(session)

    def _run_stream(
        self, session: _StreamSession, *, adapter: Any = None, encoder: Any = None
    ) -> None:
        self.send_response(200)
        content_type = encoder.get_content_type() if encoder is not None else "text/event-stream"
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        self.wfile.write(b": connected\n\n")
        self.wfile.flush()
        while not session.stopped.is_set():
            try:
                event = session.events.get(timeout=15)
            except queue.Empty:
                if session.stopped.is_set():
                    break
                self.wfile.write(b": ping\n\n")
                self.wfile.flush()
                continue
            if session.stopped.is_set():
                break

            if adapter is None:
                payload = json.dumps(event, default=str)
                self.wfile.write(f"data: {payload}\n\n".encode())
                self.wfile.flush()
                continue

            for agui_event in adapter.adapt(event):
                if session.stopped.is_set():
                    break
                try:
                    encoded = encoder.encode(agui_event)
                except (TypeError, ValueError):
                    continue
                self.wfile.write(encoded.encode("utf-8"))
                self.wfile.flush()


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
    with _lifecycle_lock:
        if _server is not None and _url is not None:
            return _url

        _token = secrets.token_urlsafe(9)
        httpd = _LiveHTTPServer((host, port), _token)
        thread = threading.Thread(target=httpd.serve_forever, name="synapsekit-live", daemon=True)
        thread.start()

        _server = httpd
        bound_host = str(httpd.server_address[0])
        bound_port = int(httpd.server_address[1])
        _url = f"http://{bound_host}:{bound_port}/?token={_token}"
        url = _url
        bus.enabled = True

        if not quiet:
            print(f"◆ SynapseKit Live → {url}")
        if open_browser:
            threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()
        return url


def stop() -> None:
    """Stop the server and disable the bus (mainly for tests)."""
    global _server, _url, _token
    with _lifecycle_lock:
        httpd = _server
        bus.enabled = False
        if httpd is not None:
            httpd.revoke_streams()
            httpd.shutdown()
            httpd.server_close()
            _server = None
            _url = None
        _token = ""


def current_url() -> str | None:
    return _url
