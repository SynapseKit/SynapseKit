from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

# A node function takes the current state and returns a partial state dict.
NodeFn = Callable[[dict[str, Any]], dict[str, Any] | Awaitable[dict[str, Any]]]


@dataclass
class Node:
    name: str
    fn: NodeFn


def agent_node(executor: Any, input_key: str = "input", output_key: str = "output") -> NodeFn:
    """Wrap an AgentExecutor as a NodeFn."""

    async def _fn(state: dict[str, Any]) -> dict[str, Any]:
        result = await executor.run(state[input_key])
        return {output_key: result}

    return _fn


def rag_node(pipeline: Any, input_key: str = "input", output_key: str = "output") -> NodeFn:
    """Wrap a RAGPipeline as a NodeFn."""

    async def _fn(state: dict[str, Any]) -> dict[str, Any]:
        result = await pipeline.ask(state[input_key])
        return {output_key: result}

    return _fn
