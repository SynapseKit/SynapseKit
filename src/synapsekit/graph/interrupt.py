"""Human-in-the-loop interrupt/resume for graph workflows."""

from __future__ import annotations

from typing import Any


class GraphInterrupt(Exception):  # noqa: N818
    """Raised by a node to pause graph execution for human review.

    The graph state is checkpointed and can be resumed after human input.

    Usage in a node function::

        def review_node(state: dict) -> dict:
            if state.get("needs_review"):
                raise GraphInterrupt(
                    message="Please review the generated content.",
                    data={"draft": state["draft"]},
                )
            return state
    """

    def __init__(self, message: str = "Graph interrupted", data: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.data = data or {}


class InterruptState:
    """Holds the state of an interrupted graph for resumption."""

    def __init__(
        self,
        graph_id: str,
        node: str,
        state: dict[str, Any],
        message: str,
        data: dict[str, Any],
        step: int,
    ) -> None:
        self.graph_id = graph_id
        self.node = node
        self.state = state
        self.message = message
        self.data = data
        self.step = step

    def __repr__(self) -> str:
        return (
            f"InterruptState(graph_id={self.graph_id!r}, node={self.node!r}, "
            f"message={self.message!r})"
        )
