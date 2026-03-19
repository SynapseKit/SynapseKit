from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import StateGraph
    from .trace import ExecutionTrace

from .state import END


def get_mermaid(graph: StateGraph) -> str:
    """Return a Mermaid flowchart string for the compiled graph."""
    lines = ["flowchart TD"]

    # Entry point arrow
    if graph._entry_point:
        lines.append(f"    __start__ --> {graph._entry_point}")

    # Static edges
    from .edge import ConditionalEdge, Edge

    for edge in graph._edges:
        if isinstance(edge, Edge):
            dst = "__end__" if edge.dst == END else edge.dst
            lines.append(f"    {edge.src} --> {dst}")
        elif isinstance(edge, ConditionalEdge):
            for label, dst in edge.mapping.items():
                dst_rendered = "__end__" if dst == END else dst
                lines.append(f"    {edge.src} -->|{label}| {dst_rendered}")

    return "\n".join(lines)


def get_mermaid_with_trace(graph: StateGraph, trace: ExecutionTrace) -> str:
    """Return a Mermaid flowchart with CSS classes for trace status.

    Nodes are styled based on their execution status:
    - ``completed`` — node finished successfully (green)
    - ``errored`` — node encountered an error (red)
    - ``skipped`` — node was not executed (gray)
    """
    base = get_mermaid(graph)
    lines = [base]

    # Categorize nodes by status
    completed_nodes: set[str] = set()
    errored_nodes: set[str] = set()

    for entry in trace.entries:
        if entry.event_type == "node_complete" and entry.node:
            completed_nodes.add(entry.node)
        elif entry.event_type == "error" and entry.node:
            errored_nodes.add(entry.node)

    # Nodes that errored should not appear as completed
    completed_nodes -= errored_nodes

    # Skipped = all graph nodes minus completed and errored
    all_nodes = set(graph._nodes.keys())
    skipped_nodes = all_nodes - completed_nodes - errored_nodes

    # Add classDef lines
    lines.append("    classDef completed fill:#90EE90,stroke:#228B22;")
    lines.append("    classDef errored fill:#FFB6C1,stroke:#DC143C;")
    lines.append("    classDef skipped fill:#D3D3D3,stroke:#808080;")

    # Add class assignments
    for node in sorted(completed_nodes):
        lines.append(f"    class {node} completed;")
    for node in sorted(errored_nodes):
        lines.append(f"    class {node} errored;")
    for node in sorted(skipped_nodes):
        lines.append(f"    class {node} skipped;")

    return "\n".join(lines)
