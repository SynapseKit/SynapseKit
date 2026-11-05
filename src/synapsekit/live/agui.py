"""Optional adapter from SynapseKit Live events to AG-UI protocol events.

The AG-UI SDK is imported only when an adapter is used, keeping the core Live
package available without the ``ag-ui`` extra.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any
from uuid import uuid4


class AGUIDependencyError(ImportError):
    """Raised when the optional AG-UI protocol package is not installed."""


_EVENT_CLASSES: dict[str, Any] | None = None


def _event_classes() -> dict[str, Any]:
    global _EVENT_CLASSES
    if _EVENT_CLASSES is None:
        try:
            from ag_ui.core import (
                RunErrorEvent,
                RunStartedEvent,
                StateDeltaEvent,
                StateSnapshotEvent,
                TextMessageContentEvent,
                TextMessageEndEvent,
                TextMessageStartEvent,
                ToolCallEndEvent,
                ToolCallStartEvent,
            )
        except ImportError as exc:
            raise AGUIDependencyError(
                "AG-UI support requires the optional dependency; install synapsekit[ag-ui]."
            ) from exc
        _EVENT_CLASSES = {
            "RunErrorEvent": RunErrorEvent,
            "RunStartedEvent": RunStartedEvent,
            "StateDeltaEvent": StateDeltaEvent,
            "StateSnapshotEvent": StateSnapshotEvent,
            "TextMessageContentEvent": TextMessageContentEvent,
            "TextMessageEndEvent": TextMessageEndEvent,
            "TextMessageStartEvent": TextMessageStartEvent,
            "ToolCallEndEvent": ToolCallEndEvent,
            "ToolCallStartEvent": ToolCallStartEvent,
        }
    return _EVENT_CLASSES


def create_event_encoder(accept: str | None = None) -> Any:
    """Create the SDK SSE encoder lazily, raising a clear error without the extra."""
    _event_classes()
    try:
        from ag_ui.encoder import EventEncoder
    except ImportError as exc:
        raise AGUIDependencyError(
            "AG-UI support requires the optional dependency; install synapsekit[ag-ui]."
        ) from exc
    if accept is None:
        return EventEncoder()
    return EventEncoder(accept=accept)


def _is_json_value(value: Any) -> bool:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError, OverflowError):
        return False
    return True


class AGUIEventAdapter:
    """Convert supported Live source facts into typed AG-UI events.

    The Live event shape does not currently carry thread, run, message, or tool
    call IDs. UUIDs generated here are transport correlation metadata only and
    are not assertions about source identities.

    Because Live replay history is bounded, a late subscriber may not receive
    the source ``run.start``. Such a subscriber receives no AG-UI data until the
    next real ``run.start``; ordinary events before it are ignored. A disconnect
    is not evidence of run completion, so this adapter does not synthesize
    ``RUN_FINISHED``.
    """

    def __init__(self) -> None:
        self.thread_id = str(uuid4())
        self.run_id = str(uuid4())
        self.last_seq: int | None = None
        self._has_run_started = False
        self._has_graph_snapshot = False

    def adapt(self, source: Mapping[str, Any]) -> list[Any]:
        """Adapt one source event; unsupported or incomplete events are ignored."""
        kind = source.get("kind")
        seq = source.get("seq")
        if isinstance(seq, int) and not isinstance(seq, bool):
            if self.last_seq is not None and seq != self.last_seq + 1:
                self._has_run_started = False
                self._has_graph_snapshot = False
            self.last_seq = seq

        if kind != "run.start" and not self._has_run_started:
            return []

        classes = _event_classes()
        attrs = source.get("attributes")
        if not isinstance(attrs, Mapping):
            attrs = {}

        events: list[Any] = []
        if kind == "run.start":
            self._has_run_started = True
            self._has_graph_snapshot = False
            self.run_id = str(uuid4())
            events.append(classes["RunStartedEvent"](threadId=self.thread_id, runId=self.run_id))
        elif kind == "tool.call":
            tool_name = attrs.get("tool")
            if isinstance(tool_name, str) and tool_name:
                tool_call_id = str(uuid4())
                events.extend(
                    [
                        classes["ToolCallStartEvent"](
                            toolCallId=tool_call_id,
                            toolCallName=tool_name,
                        ),
                        classes["ToolCallEndEvent"](toolCallId=tool_call_id),
                    ]
                )
        elif kind == "span" and source.get("name") == "llm.generate":
            output = attrs.get("llm.output")
            if isinstance(output, str) and output and source.get("status") != "error":
                message_id = str(uuid4())
                events.extend(
                    [
                        classes["TextMessageStartEvent"](messageId=message_id, role="assistant"),
                        classes["TextMessageContentEvent"](messageId=message_id, delta=output),
                        classes["TextMessageEndEvent"](messageId=message_id),
                    ]
                )
        elif kind == "graph.snapshot":
            nodes, edges = attrs.get("nodes"), attrs.get("edges")
            if isinstance(nodes, list) and isinstance(edges, list):
                graph = {"nodes": nodes, "edges": edges}
                if _is_json_value(graph):
                    if self._has_graph_snapshot:
                        events.append(
                            classes["StateDeltaEvent"](
                                delta=[{"op": "replace", "path": "/graph", "value": graph}]
                            )
                        )
                    else:
                        events.append(classes["StateSnapshotEvent"](snapshot={"graph": graph}))
                        self._has_graph_snapshot = True

        # Instrumented errors can include a traceback. The status is a supported
        # source fact; details are intentionally omitted to avoid leaking paths,
        # prompts, or secrets through the UI protocol.
        if kind in {"span", "tool.call"} and source.get("status") == "error":
            events.append(
                classes["RunErrorEvent"](
                    message="A SynapseKit Live source event reported an error."
                )
            )
        return events

    def adapt_many(self, sources: Iterable[Mapping[str, Any]]) -> list[Any]:
        """Adapt a batch in source ``seq`` order, preserving input order on ties."""
        indexed = list(enumerate(sources))

        def sort_key(item: tuple[int, Mapping[str, Any]]) -> tuple[int, int, int]:
            index, source = item
            seq = source.get("seq")
            if isinstance(seq, int) and not isinstance(seq, bool):
                return 0, seq, index
            return 1, index, index

        adapted: list[Any] = []
        for _, source in sorted(indexed, key=sort_key):
            adapted.extend(self.adapt(source))
        return adapted
