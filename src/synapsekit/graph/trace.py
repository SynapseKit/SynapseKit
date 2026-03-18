"""Execution tracing for graph workflows."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .streaming import EventHooks, GraphEvent


@dataclass
class TraceEntry:
    """A single entry in an execution trace."""

    event_type: str
    node: str | None = None
    timestamp: float = 0.0
    duration_ms: float | None = None
    data: dict[str, Any] | None = None


class ExecutionTrace:
    """Collects graph execution events into a structured trace.

    Hook into the existing :class:`EventHooks` system to record every event
    with timestamps and durations for debugging / observability.

    Usage::

        trace = ExecutionTrace()
        hooks = trace.hook(EventHooks())
        result = await compiled.run(state, hooks=hooks)
        print(trace.summary())
    """

    def __init__(self) -> None:
        self._entries: list[TraceEntry] = []
        self._node_starts: dict[str, float] = {}
        self._trace_start: float | None = None

    def hook(self, event_hooks: EventHooks) -> EventHooks:
        """Register trace callbacks on all event types. Returns the same EventHooks."""

        def _on_node_start(event: GraphEvent) -> None:
            now = time.monotonic()
            if self._trace_start is None:
                self._trace_start = now
            if event.node:
                self._node_starts[event.node] = now
            self._entries.append(
                TraceEntry(
                    event_type="node_start",
                    node=event.node,
                    timestamp=now,
                    data=event.data,
                )
            )

        def _on_node_complete(event: GraphEvent) -> None:
            now = time.monotonic()
            if self._trace_start is None:
                self._trace_start = now
            duration = None
            if event.node and event.node in self._node_starts:
                duration = (now - self._node_starts.pop(event.node)) * 1000
            self._entries.append(
                TraceEntry(
                    event_type="node_complete",
                    node=event.node,
                    timestamp=now,
                    duration_ms=duration,
                    data=event.data,
                )
            )

        def _on_wave_start(event: GraphEvent) -> None:
            now = time.monotonic()
            if self._trace_start is None:
                self._trace_start = now
            self._entries.append(
                TraceEntry(
                    event_type="wave_start",
                    node=event.node,
                    timestamp=now,
                    data=event.data,
                )
            )

        def _on_wave_complete(event: GraphEvent) -> None:
            now = time.monotonic()
            if self._trace_start is None:
                self._trace_start = now
            self._entries.append(
                TraceEntry(
                    event_type="wave_complete",
                    node=event.node,
                    timestamp=now,
                    data=event.data,
                )
            )

        def _on_error(event: GraphEvent) -> None:
            now = time.monotonic()
            if self._trace_start is None:
                self._trace_start = now
            self._entries.append(
                TraceEntry(
                    event_type="error",
                    node=event.node,
                    timestamp=now,
                    data=event.data,
                )
            )

        event_hooks.on_node_start(_on_node_start)
        event_hooks.on_node_complete(_on_node_complete)
        event_hooks.on_wave_start(_on_wave_start)
        event_hooks.on_wave_complete(_on_wave_complete)
        event_hooks.on_error(_on_error)
        return event_hooks

    @property
    def entries(self) -> list[TraceEntry]:
        """All recorded trace entries."""
        return list(self._entries)

    @property
    def total_duration_ms(self) -> float:
        """Total wall-clock duration from first to last event, in milliseconds."""
        if not self._entries:
            return 0.0
        first = self._entries[0].timestamp
        last = self._entries[-1].timestamp
        return (last - first) * 1000

    @property
    def node_durations(self) -> dict[str, float]:
        """Map of node name to execution duration in milliseconds."""
        durations: dict[str, float] = {}
        for entry in self._entries:
            if entry.event_type == "node_complete" and entry.node and entry.duration_ms is not None:
                durations[entry.node] = entry.duration_ms
        return durations

    def summary(self) -> str:
        """Human-readable execution summary."""
        if not self._entries:
            return "No events recorded."

        lines = [f"Execution trace ({len(self._entries)} events, {self.total_duration_ms:.1f}ms):"]
        for entry in self._entries:
            node_str = f" [{entry.node}]" if entry.node else ""
            dur_str = f" ({entry.duration_ms:.1f}ms)" if entry.duration_ms is not None else ""
            lines.append(f"  {entry.event_type}{node_str}{dur_str}")
        return "\n".join(lines)

    def to_dict(self) -> list[dict[str, Any]]:
        """JSON-serializable list of trace entries."""
        result = []
        for entry in self._entries:
            d: dict[str, Any] = {"event_type": entry.event_type}
            if entry.node is not None:
                d["node"] = entry.node
            d["timestamp"] = entry.timestamp
            if entry.duration_ms is not None:
                d["duration_ms"] = entry.duration_ms
            if entry.data is not None:
                d["data"] = entry.data
            result.append(d)
        return result
