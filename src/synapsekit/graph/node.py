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


def llm_node(
    llm: Any,
    input_key: str = "input",
    output_key: str = "output",
    stream: bool = False,
) -> NodeFn:
    """Wrap a BaseLLM as a NodeFn, optionally with token-level streaming.

    Args:
        llm: A ``BaseLLM`` instance.
        input_key: State key to read the prompt from.
        output_key: State key to write the response to.
        stream: If ``True``, return a ``__stream__`` key for token-level
            streaming via ``CompiledGraph.stream_tokens()``.

    Usage::

        graph.add_node("llm", llm_node(llm, stream=True))
        async for event in compiled.stream_tokens(state):
            if event["type"] == "token":
                print(event["token"], end="")
    """

    async def _fn(state: dict[str, Any]) -> dict[str, Any]:
        prompt = state[input_key]
        if stream:
            return {
                "__stream__": llm.stream(prompt),
                "__stream_key__": output_key,
            }
        result = await llm.generate(prompt)
        return {output_key: result}

    return _fn
