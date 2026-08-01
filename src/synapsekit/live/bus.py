"""Zero-dependency in-process event bus — the single stream SynapseKit Live reads.

Every observability signal (span, LLM call, retrieval, tool call, cost, decision)
is published here as a plain ``dict``. Subscribers — the dashboard's SSE stream,
and optionally the Prometheus/OTEL exporters — consume the same stream.

Design goals:
  * **No third-party dependencies** — stdlib only (``threading`` / ``queue`` / ``collections``).
  * **Zero overhead when off** — :meth:`publish` returns immediately if Live is
    disabled and nothing is subscribed, so instrumentation is free in production.
  * **Never blocks the agent** — a bounded per-subscriber queue drops on backpressure.
  * **Late joiners see the run** — a bounded history buffer replays recent events
    to a browser that connects mid-run.
"""

from __future__ import annotations

import contextlib
import queue
import threading
import time
from collections import deque
from typing import Any

# Max events replayed to a newly-connected dashboard, and max queued per client.
_HISTORY_MAX = 500
_CLIENT_QUEUE_MAX = 2000


class EventBus:
    """Thread-safe fan-out of run events to any number of subscribers."""

    def __init__(self, history: int = _HISTORY_MAX) -> None:
        self._lock = threading.Lock()
        self._subscribers: set[queue.Queue[dict[str, Any]]] = set()
        self._history: deque[dict[str, Any]] = deque(maxlen=history)
        self._seq = 0
        # Live is off by default; the server flips this on when it starts.
        self.enabled = False

    # -- producer side ----------------------------------------------------
    def publish(self, event: dict[str, Any]) -> None:
        """Publish one event. No-op (a single attribute read) when Live is off."""
        if not self.enabled:
            return
        with self._lock:
            self._seq += 1
            enriched = {"seq": self._seq, "ts": time.time(), **event}
            self._history.append(enriched)
            subscribers = tuple(self._subscribers)
        for q in subscribers:
            # Drop on backpressure — telemetry must never stall the agent.
            with contextlib.suppress(queue.Full):
                q.put_nowait(enriched)

    # -- consumer side ----------------------------------------------------
    def subscribe(self) -> queue.Queue[dict[str, Any]]:
        """Register a subscriber, pre-loaded with recent history for late joiners."""
        q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=_CLIENT_QUEUE_MAX)
        with self._lock:
            for past in self._history:
                try:
                    q.put_nowait(past)
                except queue.Full:
                    break
            self._subscribers.add(q)
        return q

    def unsubscribe(self, q: queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            self._subscribers.discard(q)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    def history(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._history)

    def clear(self) -> None:
        with self._lock:
            self._history.clear()
            self._seq = 0


# Process-wide singleton every SynapseKit component publishes to.
bus = EventBus()


def publish(kind: str, **fields: Any) -> None:
    """Convenience publisher: ``publish("tool.call", tool="notion", ok=True)``."""
    bus.publish({"kind": kind, **fields})
