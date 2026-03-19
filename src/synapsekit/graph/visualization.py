from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .compiled import CompiledGraph
    from .trace import ExecutionTrace

from .mermaid import get_mermaid, get_mermaid_with_trace


class GraphVisualizer:
    """Visualization and replay utilities for compiled graphs.

    Provides ASCII timeline rendering, Mermaid diagram generation,
    step-by-step replay, and standalone HTML export.
    """

    def __init__(self, compiled_graph: CompiledGraph) -> None:
        self._compiled = compiled_graph

    def render_trace(self, trace: ExecutionTrace) -> str:
        """Render an ASCII timeline with wave grouping and durations."""
        lines: list[str] = []
        wave_num = 0
        total_ms = 0.0

        # Build lookup for node durations and error status
        node_durations: dict[str, float] = {}
        errored_nodes: set[str] = set()
        for entry in trace.entries:
            if entry.event_type == "node_complete" and entry.node and entry.duration_ms is not None:
                node_durations[entry.node] = entry.duration_ms
            elif entry.event_type == "error" and entry.node:
                errored_nodes.add(entry.node)

        # Walk entries to group by wave
        for entry in trace.entries:
            if entry.event_type == "wave_start":
                wave_num += 1
                lines.append(f"Wave {wave_num}:")
            elif entry.event_type == "node_complete" and entry.node:
                dur = entry.duration_ms if entry.duration_ms is not None else 0.0
                total_ms += dur
                error_marker = "[!] " if entry.node in errored_nodes else ""
                lines.append(f"  {error_marker}[{entry.node}] {dur:.1f}ms")
            elif entry.event_type == "error" and entry.node:
                # Error entries that are not also node_complete
                if entry.node not in node_durations:
                    lines.append(f"  [!] [{entry.node}] error")

        lines.append(f"Total: {total_ms:.1f}ms")
        return "\n".join(lines)

    def render_mermaid(self, trace: ExecutionTrace | None = None) -> str:
        """Return a Mermaid flowchart string, optionally styled with trace data."""
        if trace is None:
            return get_mermaid(self._compiled._graph)
        return get_mermaid_with_trace(self._compiled._graph, trace)

    def replay_steps(self, trace: ExecutionTrace) -> list[dict[str, Any]]:
        """Return list of step dicts from trace entries.

        Only includes ``node_complete`` and ``error`` entries.
        Each dict contains: node, duration_ms, wave, status, data.
        """
        steps: list[dict[str, Any]] = []
        wave_num = 0

        for entry in trace.entries:
            if entry.event_type == "wave_start":
                wave_num += 1
            elif entry.event_type == "node_complete" and entry.node:
                steps.append({
                    "node": entry.node,
                    "duration_ms": entry.duration_ms if entry.duration_ms is not None else 0.0,
                    "wave": wave_num,
                    "status": "completed",
                    "data": entry.data,
                })
            elif entry.event_type == "error" and entry.node:
                steps.append({
                    "node": entry.node,
                    "duration_ms": entry.duration_ms if entry.duration_ms is not None else 0.0,
                    "wave": wave_num,
                    "status": "errored",
                    "data": entry.data,
                })

        return steps

    def to_html(self, trace: ExecutionTrace | None = None) -> str:
        """Return a standalone HTML string with an embedded Mermaid diagram.

        If *trace* is provided the ASCII timeline is included below the diagram.
        """
        mermaid_code = self.render_mermaid(trace)

        timeline_section = ""
        if trace is not None:
            ascii_timeline = self.render_trace(trace)
            timeline_section = (
                f'<h2>Execution Timeline</h2>\n'
                f'<pre>{ascii_timeline}</pre>'
            )

        return (
            "<!DOCTYPE html>\n"
            "<html>\n"
            "<head>\n"
            '  <meta charset="utf-8">\n'
            "  <title>Graph Visualization</title>\n"
            '  <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>\n'
            "  <script>mermaid.initialize({startOnLoad: true});</script>\n"
            "</head>\n"
            "<body>\n"
            '<div class="mermaid">\n'
            f"{mermaid_code}\n"
            "</div>\n"
            f"{timeline_section}\n"
            "</body>\n"
            "</html>"
        )
