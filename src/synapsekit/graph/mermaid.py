from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import StateGraph

from .state import END


def get_mermaid(graph: "StateGraph") -> str:
    """Return a Mermaid flowchart string for the compiled graph."""
    lines = ["flowchart TD"]

    # Entry point arrow
    if graph._entry_point:
        lines.append(f"    __start__ --> {graph._entry_point}")

    # Static edges
    from .edge import Edge, ConditionalEdge
    for edge in graph._edges:
        if isinstance(edge, Edge):
            dst = "__end__" if edge.dst == END else edge.dst
            lines.append(f"    {edge.src} --> {dst}")
        elif isinstance(edge, ConditionalEdge):
            for label, dst in edge.mapping.items():
                dst_rendered = "__end__" if dst == END else dst
                lines.append(f"    {edge.src} -->|{label}| {dst_rendered}")

    return "\n".join(lines)
