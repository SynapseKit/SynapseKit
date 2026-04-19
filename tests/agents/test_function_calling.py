"""Tests for FunctionCallingAgent."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from synapsekit.agents.base import BaseTool, ToolResult
from synapsekit.agents.function_calling import FunctionCallingAgent
from synapsekit.llm.base import BaseLLM, LLMConfig

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


class AddTool(BaseTool):
    name = "add"
    description = "Add two numbers."
    parameters = {
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "required": ["a", "b"],
    }

    async def run(self, a=0, b=0, **kwargs) -> ToolResult:
        return ToolResult(output=str(a + b))


def make_fc_llm(responses):
    """LLM with call_with_tools returning responses in sequence."""
    llm = MagicMock()
    llm.call_with_tools = AsyncMock(side_effect=responses)
    return llm


def make_no_fc_llm():
    """LLM that does not override call_with_tools — simulates Ollama etc."""

    class _NoFCLLM(BaseLLM):
        async def stream(self, prompt, **kw):
            yield "x"

    return _NoFCLLM(LLMConfig(model="test", api_key="test", provider="test"))


@dataclass
class _MemoryRecord:
    content: str
    memory_type: str = "semantic"


# ------------------------------------------------------------------ #
# FunctionCallingAgent tests
# ------------------------------------------------------------------ #


class TestFunctionCallingAgent:
    @pytest.mark.asyncio
    async def test_direct_answer_no_tool_call(self):
        responses = [{"content": "Paris is the capital of France.", "tool_calls": None}]
        llm = make_fc_llm(responses)
        agent = FunctionCallingAgent(llm=llm, tools=[])
        result = await agent.run("What is the capital of France?")
        assert result == "Paris is the capital of France."

    @pytest.mark.asyncio
    async def test_single_tool_call(self):
        responses = [
            {
                "content": None,
                "tool_calls": [{"id": "tc1", "name": "add", "arguments": {"a": 3, "b": 4}}],
            },
            {"content": "The answer is 7.", "tool_calls": None},
        ]
        llm = make_fc_llm(responses)
        agent = FunctionCallingAgent(llm=llm, tools=[AddTool()])
        result = await agent.run("What is 3 + 4?")
        assert result == "The answer is 7."

    @pytest.mark.asyncio
    async def test_memory_records_tool_calls(self):
        responses = [
            {
                "content": None,
                "tool_calls": [{"id": "tc1", "name": "add", "arguments": {"a": 1, "b": 2}}],
            },
            {"content": "Done.", "tool_calls": None},
        ]
        llm = make_fc_llm(responses)
        agent = FunctionCallingAgent(llm=llm, tools=[AddTool()])
        await agent.run("1 + 2?")
        assert len(agent.memory) == 1
        assert agent.memory.steps[0].action == "add"
        assert agent.memory.steps[0].observation == "3"

    @pytest.mark.asyncio
    async def test_unknown_tool_handled(self):
        responses = [
            {
                "content": None,
                "tool_calls": [{"id": "tc1", "name": "nonexistent", "arguments": {}}],
            },
            {"content": "Could not complete.", "tool_calls": None},
        ]
        llm = make_fc_llm(responses)
        agent = FunctionCallingAgent(llm=llm, tools=[])
        result = await agent.run("do something")
        assert result == "Could not complete."
        assert "Error" in agent.memory.steps[0].observation

    @pytest.mark.asyncio
    async def test_max_iterations_respected(self):
        # Always returns tool calls — never a final answer
        repeated = {
            "content": None,
            "tool_calls": [{"id": "tc0", "name": "add", "arguments": {"a": 1, "b": 1}}],
        }
        llm = make_fc_llm([repeated] * 20)
        agent = FunctionCallingAgent(llm=llm, tools=[AddTool()], max_iterations=3)
        result = await agent.run("keep going")
        assert "unable" in result.lower()

    @pytest.mark.asyncio
    async def test_raises_if_llm_has_no_call_with_tools(self):
        llm = make_no_fc_llm()
        agent = FunctionCallingAgent(llm=llm, tools=[])
        with pytest.raises(RuntimeError, match="does not support native function calling"):
            await agent.run("test")

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        responses = [{"content": "The answer is 42.", "tool_calls": None}]
        llm = make_fc_llm(responses)
        agent = FunctionCallingAgent(llm=llm, tools=[])
        tokens = []
        async for t in agent.stream("What is the answer?"):
            tokens.append(t)
        assert "42." in " ".join(tokens)

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_per_step(self):
        responses = [
            {
                "content": None,
                "tool_calls": [
                    {"id": "tc1", "name": "add", "arguments": {"a": 1, "b": 2}},
                    {"id": "tc2", "name": "add", "arguments": {"a": 3, "b": 4}},
                ],
            },
            {"content": "Results: 3 and 7.", "tool_calls": None},
        ]
        llm = make_fc_llm(responses)
        agent = FunctionCallingAgent(llm=llm, tools=[AddTool()])
        result = await agent.run("compute both")
        assert result == "Results: 3 and 7."
        assert len(agent.memory) == 2

    @pytest.mark.asyncio
    async def test_persistent_memory_recalled_and_injected(self):
        persistent = MagicMock()
        persistent.recall = AsyncMock(return_value=[_MemoryRecord(content="Use metric units")])
        persistent.store = AsyncMock()

        responses = [{"content": "Done", "tool_calls": None}]
        llm = make_fc_llm(responses)
        agent = FunctionCallingAgent(llm=llm, tools=[], memory=persistent, agent_id="u1")

        result = await agent.run("How should I format output?")
        assert result == "Done"

        persistent.recall.assert_awaited_once_with(
            agent_id="u1", query="How should I format output?", top_k=5
        )
        messages = llm.call_with_tools.call_args.args[0]
        assert "Relevant persistent memories" in messages[0]["content"]
        assert "Use metric units" in messages[0]["content"]

    @pytest.mark.asyncio
    async def test_persistent_memory_stores_episodic_run(self):
        persistent = MagicMock()
        persistent.recall = AsyncMock(return_value=[])
        persistent.store = AsyncMock()

        responses = [{"content": "final", "tool_calls": None}]
        llm = make_fc_llm(responses)
        agent = FunctionCallingAgent(llm=llm, tools=[], memory=persistent, agent_id="u2")

        await agent.run("question")
        persistent.store.assert_awaited_once()
        payload = persistent.store.await_args.kwargs
        assert payload["agent_id"] == "u2"
        assert payload["memory_type"] == "episodic"
        assert "question" in payload["content"]
        assert "final" in payload["content"]
