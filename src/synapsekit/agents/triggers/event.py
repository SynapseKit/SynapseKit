"""Webhook-triggered agent execution via an aiohttp HTTP server."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import inspect
import json
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from .cron import TriggerResult

_MISSING = object()


class EventTrigger:
    """Receive HTTP webhook payloads and dispatch them to an agent function.

    Install: ``pip install synapsekit[event-trigger]``  (requires ``aiohttp>=3.9``).

    Parameters
    ----------
    agent_fn:
        Async callable ``(payload: dict) -> str`` invoked for each request.
    host:
        Bind address for the HTTP server (default ``"127.0.0.1"``).
    port:
        TCP port to listen on (default ``8765``).
    path:
        URL path that accepts POST requests (default ``"/webhook"``).
    secret:
        Optional HMAC-SHA256 secret.  When set, incoming requests must supply
        a ``X-Webhook-Secret`` header whose value equals
        ``hmac.new(secret, body, sha256).hexdigest()``.
    result_sink:
        Optional async/sync callable receiving a :class:`TriggerResult` after
        each invocation.
    """

    def __init__(
        self,
        agent_fn: Callable[[dict[str, Any]], Awaitable[str]],
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        path: str = "/webhook",
        secret: str | None = None,
        result_sink: Callable[[TriggerResult], Any] | None = None,
    ) -> None:
        self._agent_fn = agent_fn
        self._host = host
        self._port = port
        self._path = path
        self._secret = secret
        self._result_sink = result_sink
        self._runner: Any = None
        self._site: Any = None
        self._stop_event: asyncio.Event = asyncio.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the aiohttp server (non-blocking)."""
        try:
            from aiohttp import web
        except ImportError:
            raise ImportError(
                "aiohttp package required: pip install synapsekit[event-trigger]"
            ) from None

        app = web.Application()
        app.router.add_post(self._path, self._handle)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()

    async def stop(self) -> None:
        """Shut down the HTTP server."""
        self._stop_event.set()
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
            self._site = None

    async def run_forever(self) -> None:
        """Start the server and block until :meth:`stop` is called."""
        await self.start()
        await self._stop_event.wait()
        await self.stop()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _verify_signature(self, body: bytes, header_value: str | None) -> bool:
        if self._secret is None:
            return True
        if not header_value:
            return False
        expected = hmac.new(self._secret.encode(), body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, header_value)

    async def _handle(self, request: Any) -> Any:
        from aiohttp import web

        body = await request.read()
        sig_header = request.headers.get("X-Webhook-Secret")

        if not self._verify_signature(body, sig_header):
            return web.Response(status=401, text="Unauthorized")

        try:
            payload: dict[str, Any] = json.loads(body) if body else {}
        except json.JSONDecodeError:
            return web.Response(status=400, text="Invalid JSON")

        scheduled_for = datetime.now(tz=timezone.utc)
        started_at = datetime.now(tz=timezone.utc)
        output: str | None = None
        error_text: str | None = None

        try:
            output = await self._agent_fn(payload)
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"

        finished_at = datetime.now(tz=timezone.utc)
        result = TriggerResult(
            scheduled_for=scheduled_for,
            started_at=started_at,
            finished_at=finished_at,
            input_text=json.dumps(payload),
            output=output,
            error=error_text,
            metadata={"host": self._host, "port": self._port, "path": self._path},
        )

        if self._result_sink is not None:
            maybe = self._result_sink(result)
            if inspect.isawaitable(maybe):
                await maybe

        if error_text:
            return web.Response(status=500, text=error_text)
        return web.Response(
            status=200,
            content_type="application/json",
            text=json.dumps({"output": output}),
        )
