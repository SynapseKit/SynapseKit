from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

# A node function takes the current state and returns a partial state dict.
NodeFn = Callable[[dict[str, Any]], dict[str, Any] | Awaitable[dict[str, Any]]]


@dataclass
class Node:
    name: str
    fn: NodeFn
    metadata: dict[str, Any] = field(default_factory=dict)


def agent_node(
    executor: Any,
    input_key: str = "input",
    output_key: str = "output",
    *,
    memory_key: str | None = None,
    agent_id_key: str | None = None,
    memory_top_k_key: str | None = None,
) -> NodeFn:
    """Wrap an AgentExecutor as a NodeFn.

    Optional state wiring for memory-aware agents:
    - ``memory_key``: state key containing persistent ``AgentMemory`` instance
    - ``agent_id_key``: state key containing per-run agent/user id
    - ``memory_top_k_key``: state key overriding recall top-k

    When provided, these values are applied to ``executor.config`` before each run.
    """

    def _configure_executor_from_state(state: dict[str, Any]) -> None:
        config = getattr(executor, "config", None)
        if config is None:
            return

        changed = False

        if (
            memory_key
            and memory_key in state
            and getattr(config, "memory", None) is not state[memory_key]
        ):
            config.memory = state[memory_key]
            changed = True

        if agent_id_key and agent_id_key in state:
            agent_id = state[agent_id_key]
            if agent_id is not None and getattr(config, "agent_id", None) != str(agent_id):
                config.agent_id = str(agent_id)
                changed = True

        if memory_top_k_key and memory_top_k_key in state:
            top_k = int(state[memory_top_k_key])
            if getattr(config, "memory_top_k", None) != top_k:
                config.memory_top_k = top_k
                changed = True

        if changed and hasattr(executor, "_build_agent"):
            executor._agent = executor._build_agent()

    async def _fn(state: dict[str, Any]) -> dict[str, Any]:
        _configure_executor_from_state(state)
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
