"""SSE and event callback streaming for graph execution."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraphEvent:
    """An event emitted during graph execution."""

    event_type: str  # "node_start", "node_complete", "wave_start", "wave_complete", "error"
    node: str | None = None
    state: dict[str, Any] | None = None
    data: dict[str, Any] | None = None

    def to_sse(self) -> str:
        """Format as a Server-Sent Event string."""
        payload: dict[str, Any] = {
            "event": self.event_type,
            "node": self.node,
        }
        if self.state is not None:
            payload["state"] = self.state
        if self.data is not None:
            payload["data"] = self.data
        return f"event: {self.event_type}\ndata: {json.dumps(payload)}\n\n"

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict."""
        result: dict[str, Any] = {"event": self.event_type}
        if self.node is not None:
            result["node"] = self.node
        if self.state is not None:
            result["state"] = self.state
        if self.data is not None:
            result["data"] = self.data
        return result

    def to_ws(self) -> str:
        """Format as JSON string for WebSocket transmission."""
        return json.dumps(self.to_dict())


# Type alias for event callbacks
EventCallback = Callable[[GraphEvent], Any]


@dataclass
class EventHooks:
    """Collection of event callbacks for graph execution.

    Usage::

        hooks = EventHooks()
        hooks.on_node_start(lambda e: print(f"Starting {e.node}"))
        hooks.on_node_complete(lambda e: print(f"Done {e.node}"))

        result = await compiled.run(state, hooks=hooks)
    """

    _callbacks: dict[str, list[EventCallback]] = field(default_factory=dict)

    def on(self, event_type: str, callback: EventCallback) -> EventHooks:
        """Register a callback for an event type."""
        self._callbacks.setdefault(event_type, []).append(callback)
        return self

    def on_node_start(self, callback: EventCallback) -> EventHooks:
        """Register a callback for node_start events."""
        return self.on("node_start", callback)

    def on_node_complete(self, callback: EventCallback) -> EventHooks:
        """Register a callback for node_complete events."""
        return self.on("node_complete", callback)

    def on_wave_start(self, callback: EventCallback) -> EventHooks:
        """Register a callback for wave_start events."""
        return self.on("wave_start", callback)

    def on_wave_complete(self, callback: EventCallback) -> EventHooks:
        """Register a callback for wave_complete events."""
        return self.on("wave_complete", callback)

    def on_error(self, callback: EventCallback) -> EventHooks:
        """Register a callback for error events."""
        return self.on("error", callback)

    async def emit(self, event: GraphEvent) -> None:
        """Emit an event to all registered callbacks."""
        import inspect

        for cb in self._callbacks.get(event.event_type, []):
            result = cb(event)
            if inspect.isawaitable(result):
                await result


async def sse_stream(
    compiled_graph: Any,
    state: dict[str, Any],
) -> AsyncGenerator[str]:
    """Stream graph execution as Server-Sent Events.

    Yields SSE-formatted strings suitable for HTTP responses::

        from starlette.responses import StreamingResponse

        async def endpoint(request):
            return StreamingResponse(
                sse_stream(compiled, {"input": "hello"}),
                media_type="text/event-stream",
            )

    Events emitted:
    - ``node_complete`` — after each node finishes, with current state
    - ``done`` — when graph execution completes
    """
    state = dict(state)
    async for event in compiled_graph.stream(state):
        graph_event = GraphEvent(
            event_type="node_complete",
            node=event.get("node"),
            state=event.get("state"),
        )
        yield graph_event.to_sse()
    yield f"event: done\ndata: {json.dumps({'state': state})}\n\n"


async def ws_stream(
    compiled_graph: Any,
    state: dict[str, Any],
    websocket: Any,
    hooks: EventHooks | None = None,
) -> dict[str, Any]:
    """Run graph and stream events over a WebSocket connection.

    Works with any WebSocket object that has a ``send_text()`` or ``send()``
    method (e.g. Starlette, FastAPI, or plain websockets).

    Usage::

        @app.websocket("/ws")
        async def endpoint(websocket):
            await websocket.accept()
            result = await ws_stream(compiled, {"input": "hello"}, websocket)

    Args:
        compiled_graph: A compiled :class:`CompiledGraph`.
        state: Initial graph state dict.
        websocket: Any object with ``send_text(str)`` or ``send(str)`` method.
        hooks: Optional extra event hooks to run alongside streaming.

    Returns:
        The final state dict after graph execution.
    """
    ws_hooks = EventHooks()
    send = getattr(websocket, "send_text", None) or websocket.send

    async def _send_event(event: GraphEvent) -> None:
        await send(event.to_ws())

    ws_hooks.on_node_start(_send_event)
    ws_hooks.on_node_complete(_send_event)
    ws_hooks.on_wave_start(_send_event)
    ws_hooks.on_wave_complete(_send_event)
    ws_hooks.on_error(_send_event)

    # Merge user-provided hooks
    if hooks:
        for event_type, cbs in hooks._callbacks.items():
            for cb in cbs:
                ws_hooks.on(event_type, cb)

    state = dict(state)
    result = await compiled_graph.run(state, hooks=ws_hooks)

    done_event = GraphEvent(event_type="done", state=result)
    await send(done_event.to_ws())

    return result
