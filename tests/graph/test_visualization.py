from __future__ import annotations

from unittest.mock import MagicMock

from synapsekit.graph.mermaid import get_mermaid_with_trace
from synapsekit.graph.trace import ExecutionTrace, TraceEntry
from synapsekit.graph.visualization import GraphVisualizer


def _make_mock_graph():
    """Create a mock CompiledGraph with a simple graph structure."""
    mock_graph = MagicMock()
    mock_graph._graph._entry_point = "a"
    mock_graph._graph._nodes = {
        "a": MagicMock(),
        "b": MagicMock(),
        "c": MagicMock(),
    }
    mock_graph._graph._edges = []  # empty edges for simplicity
    return mock_graph


def _make_trace(entries):
    trace = ExecutionTrace()
    trace._entries = entries
    return trace


# ------------------------------------------------------------------ #
# get_mermaid_with_trace
# ------------------------------------------------------------------ #


class TestGetMermaidWithTrace:
    def test_completed_nodes_styled(self):
        mock_graph = _make_mock_graph()
        trace = _make_trace([
            TraceEntry(event_type="node_complete", node="a", timestamp=1.0, duration_ms=10.0),
            TraceEntry(event_type="node_complete", node="b", timestamp=2.0, duration_ms=5.0),
        ])
        result = get_mermaid_with_trace(mock_graph._graph, trace)
        assert "class a completed;" in result
        assert "class b completed;" in result
        assert "classDef completed fill:#90EE90,stroke:#228B22;" in result

    def test_errored_nodes_styled(self):
        mock_graph = _make_mock_graph()
        trace = _make_trace([
            TraceEntry(event_type="node_complete", node="a", timestamp=1.0, duration_ms=10.0),
            TraceEntry(event_type="error", node="b", timestamp=2.0),
        ])
        result = get_mermaid_with_trace(mock_graph._graph, trace)
        assert "class b errored;" in result
        assert "classDef errored fill:#FFB6C1,stroke:#DC143C;" in result
        # b should not be marked completed
        assert "class b completed;" not in result

    def test_skipped_nodes_styled(self):
        mock_graph = _make_mock_graph()
        trace = _make_trace([
            TraceEntry(event_type="node_complete", node="a", timestamp=1.0, duration_ms=10.0),
        ])
        result = get_mermaid_with_trace(mock_graph._graph, trace)
        assert "class a completed;" in result
        # b and c were never executed
        assert "class b skipped;" in result
        assert "class c skipped;" in result
        assert "classDef skipped fill:#D3D3D3,stroke:#808080;" in result


# ------------------------------------------------------------------ #
# GraphVisualizer
# ------------------------------------------------------------------ #


class TestGraphVisualizer:
    def test_render_trace_ascii(self):
        mock_graph = _make_mock_graph()
        viz = GraphVisualizer(mock_graph)
        trace = _make_trace([
            TraceEntry(event_type="wave_start", timestamp=0.0, data={"wave": ["a", "b"], "step": 1}),
            TraceEntry(event_type="node_complete", node="a", timestamp=1.0, duration_ms=12.3),
            TraceEntry(event_type="node_complete", node="b", timestamp=1.1, duration_ms=8.1),
            TraceEntry(event_type="wave_complete", timestamp=1.2, data={"wave": ["a", "b"], "step": 1}),
            TraceEntry(event_type="wave_start", timestamp=1.3, data={"wave": ["c"], "step": 2}),
            TraceEntry(event_type="node_complete", node="c", timestamp=2.0, duration_ms=45.2),
            TraceEntry(event_type="wave_complete", timestamp=2.1, data={"wave": ["c"], "step": 2}),
        ])
        result = viz.render_trace(trace)
        assert "Wave 1:" in result
        assert "[a] 12.3ms" in result
        assert "[b] 8.1ms" in result
        assert "Wave 2:" in result
        assert "[c] 45.2ms" in result
        assert "Total: 65.6ms" in result

    def test_render_trace_errored_node(self):
        mock_graph = _make_mock_graph()
        viz = GraphVisualizer(mock_graph)
        trace = _make_trace([
            TraceEntry(event_type="wave_start", timestamp=0.0),
            TraceEntry(event_type="error", node="a", timestamp=1.0),
            TraceEntry(event_type="node_complete", node="a", timestamp=1.0, duration_ms=5.0),
            TraceEntry(event_type="wave_complete", timestamp=1.1),
        ])
        result = viz.render_trace(trace)
        assert "[!] [a] 5.0ms" in result

    def test_render_mermaid_no_trace(self):
        mock_graph = _make_mock_graph()
        viz = GraphVisualizer(mock_graph)
        result = viz.render_mermaid()
        assert "flowchart TD" in result
        assert "__start__ --> a" in result
        # Should not have classDef lines
        assert "classDef" not in result

    def test_render_mermaid_with_trace(self):
        mock_graph = _make_mock_graph()
        viz = GraphVisualizer(mock_graph)
        trace = _make_trace([
            TraceEntry(event_type="node_complete", node="a", timestamp=1.0, duration_ms=10.0),
        ])
        result = viz.render_mermaid(trace)
        assert "flowchart TD" in result
        assert "classDef completed" in result
        assert "class a completed;" in result

    def test_replay_steps(self):
        mock_graph = _make_mock_graph()
        viz = GraphVisualizer(mock_graph)
        trace = _make_trace([
            TraceEntry(event_type="wave_start", timestamp=0.0),
            TraceEntry(event_type="node_complete", node="a", timestamp=1.0, duration_ms=12.3, data={"key": "val"}),
            TraceEntry(event_type="wave_complete", timestamp=1.1),
            TraceEntry(event_type="wave_start", timestamp=1.2),
            TraceEntry(event_type="error", node="b", timestamp=2.0, data={"error": "fail"}),
            TraceEntry(event_type="wave_complete", timestamp=2.1),
        ])
        steps = viz.replay_steps(trace)
        assert len(steps) == 2
        assert steps[0] == {
            "node": "a",
            "duration_ms": 12.3,
            "wave": 1,
            "status": "completed",
            "data": {"key": "val"},
        }
        assert steps[1] == {
            "node": "b",
            "duration_ms": 0.0,
            "wave": 2,
            "status": "errored",
            "data": {"error": "fail"},
        }

    def test_to_html_contains_mermaid(self):
        mock_graph = _make_mock_graph()
        viz = GraphVisualizer(mock_graph)
        html = viz.to_html()
        assert "mermaid.min.js" in html
        assert "flowchart TD" in html
        assert "<html>" in html
        # No trace, so no timeline
        assert "Execution Timeline" not in html

    def test_to_html_with_trace(self):
        mock_graph = _make_mock_graph()
        viz = GraphVisualizer(mock_graph)
        trace = _make_trace([
            TraceEntry(event_type="wave_start", timestamp=0.0),
            TraceEntry(event_type="node_complete", node="a", timestamp=1.0, duration_ms=10.0),
            TraceEntry(event_type="wave_complete", timestamp=1.1),
        ])
        html = viz.to_html(trace)
        assert "mermaid.min.js" in html
        assert "Execution Timeline" in html
        assert "Wave 1:" in html
        assert "[a] 10.0ms" in html
