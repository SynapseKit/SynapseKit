"""Distributed tracing for multi-agent workflows."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceSpan:
    """A span in a distributed trace."""

    trace_id: str
    span_id: str
    name: str
    parent_span_id: str | None = None
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    events: list[dict[str, Any]] = field(default_factory=list)

    def end(self) -> None:
        self.end_time = time.time()

    @property
    def duration_ms(self) -> float:
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append({"name": name, "time": time.time(), "attributes": attributes or {}})

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "status": self.status,
            "events": self.events,
        }


class DistributedTracer:
    """Trace execution across multiple agents with parent-child relationships.

    Usage::
        tracer = DistributedTracer()

        # Start a root span for the supervisor
        root = tracer.start_span("supervisor.run")

        # Create child spans for worker agents
        worker_span = tracer.start_span("worker.research", parent=root)
        # ... worker does work ...
        worker_span.end()

        root.end()

        # Get full trace
        trace = tracer.get_trace()
    """

    def __init__(self, trace_id: str | None = None) -> None:
        self._trace_id = trace_id or uuid.uuid4().hex[:16]
        self._spans: list[TraceSpan] = []

    @property
    def trace_id(self) -> str:
        return self._trace_id

    def start_span(
        self,
        name: str,
        parent: TraceSpan | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> TraceSpan:
        span = TraceSpan(
            trace_id=self._trace_id,
            span_id=uuid.uuid4().hex[:16],
            name=name,
            parent_span_id=parent.span_id if parent else None,
            attributes=attributes or {},
        )
        self._spans.append(span)
        return span

    def get_trace(self) -> list[dict[str, Any]]:
        """Get all spans as a list of dicts."""
        return [s.to_dict() for s in self._spans]

    def get_root_spans(self) -> list[TraceSpan]:
        """Get spans with no parent."""
        return [s for s in self._spans if s.parent_span_id is None]

    def get_children(self, span: TraceSpan) -> list[TraceSpan]:
        """Get child spans of a given span."""
        return [s for s in self._spans if s.parent_span_id == span.span_id]

    @property
    def total_duration_ms(self) -> float:
        roots = self.get_root_spans()
        if not roots:
            return 0.0
        return sum(r.duration_ms for r in roots)

    @property
    def span_count(self) -> int:
        return len(self._spans)

    def clear(self) -> None:
        self._spans.clear()
