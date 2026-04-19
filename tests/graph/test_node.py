"""Tests for graph node helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from synapsekit.agents.executor import AgentConfig, AgentExecutor
from synapsekit.graph.node import agent_node
from synapsekit.memory import AgentMemory as PersistentAgentMemory


@pytest.mark.asyncio
async def test_agent_node_basic_wrapper():
    class _Executor:
        async def run(self, query: str) -> str:
            return f"answer:{query}"

    fn = agent_node(_Executor(), input_key="q", output_key="a")
    result = await fn({"q": "hello"})
    assert result == {"a": "answer:hello"}


@pytest.mark.asyncio
async def test_agent_node_memory_wiring_into_executor_config():
    llm = MagicMock()
    llm.generate_with_messages = AsyncMock(return_value="Thought: done\nFinal Answer: ok")

    executor = AgentExecutor(
        AgentConfig(
            llm=llm,
            tools=[],
            agent_type="react",
        )
    )

    persistent = PersistentAgentMemory(backend="memory")
    await persistent.store(
        agent_id="alice",
        content="Alice prefers tea over coffee",
        memory_type="semantic",
    )

    fn = agent_node(
        executor,
        input_key="q",
        output_key="a",
        memory_key="memory",
        agent_id_key="agent_id",
        memory_top_k_key="top_k",
    )

    result = await fn(
        {
            "q": "What drink should I suggest?",
            "memory": persistent,
            "agent_id": "alice",
            "top_k": 3,
        }
    )
    assert result == {"a": "ok"}

    assert executor.config.memory is persistent
    assert executor.config.agent_id == "alice"
    assert executor.config.memory_top_k == 3

    messages = llm.generate_with_messages.call_args.args[0]
    assert "Relevant persistent memories" in messages[0]["content"]
    assert "Alice prefers tea over coffee" in messages[0]["content"]
