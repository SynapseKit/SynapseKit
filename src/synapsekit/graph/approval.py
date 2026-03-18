"""Approval node — gate graph execution on human approval."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .interrupt import GraphInterrupt
from .node import NodeFn


def approval_node(
    approval_key: str = "approved",
    message: str | Callable[[dict[str, Any]], str] = "Approval required to continue.",
    data: dict[str, Any] | None = None,
) -> NodeFn:
    """Factory that returns a node function gating on human approval.

    If ``state[approval_key]`` is truthy the node passes through unchanged.
    Otherwise it raises :class:`GraphInterrupt` so the graph pauses for
    human review.

    Args:
        approval_key: State key to check for approval (default ``"approved"``).
        message: Static string or callable ``(state) -> str`` for the
            interrupt message shown to the human reviewer.
        data: Optional extra data to attach to the :class:`GraphInterrupt`.

    Usage::

        graph.add_node("gate", approval_node(
            approval_key="human_ok",
            message=lambda s: f"Please approve: {s.get('draft', '')[:100]}",
        ))
    """
    extra_data = data or {}

    async def _fn(state: dict[str, Any]) -> dict[str, Any]:
        if state.get(approval_key):
            return state

        msg = message(state) if callable(message) else message
        raise GraphInterrupt(message=msg, data={**extra_data, **state})

    return _fn
