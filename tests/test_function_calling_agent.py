"""Tests for FunctionCallingAgent."""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from synapsekit.agents.function_calling import FunctionCallingAgent
from synapsekit.agents.base import BaseTool, ToolResult


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
    """LLM without call_with_tools — simulates Ollama/Gemini etc."""
    llm = MagicMock(spec=[])  # no call_with_tools attribute
    return llm


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
