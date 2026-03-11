"""Tests for ReActAgent."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from synapsekit.agents.base import BaseTool, ToolResult
from synapsekit.agents.react import ReActAgent, _parse_action, _parse_final_answer, _parse_thought

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


class EchoTool(BaseTool):
    name = "echo"
    description = "Echoes the input back."
    parameters = {
        "type": "object",
        "properties": {"input": {"type": "string"}},
        "required": ["input"],
    }

    async def run(self, input: str = "", **kwargs) -> ToolResult:
        return ToolResult(output=f"ECHO: {input}")


class ErrorTool(BaseTool):
    name = "bad_tool"
    description = "Always raises."
    parameters = {}

    async def run(self, **kwargs) -> ToolResult:
        raise RuntimeError("tool exploded")


def make_mock_llm(responses):
    """LLM that returns responses in sequence."""
    llm = MagicMock()
    llm.generate_with_messages = AsyncMock(side_effect=responses)
    return llm


# ------------------------------------------------------------------ #
# Parser unit tests
# ------------------------------------------------------------------ #


class TestParsers:
    def test_parse_final_answer(self):
        text = "Thought: done\nFinal Answer: The result is 42"
        assert _parse_final_answer(text) == "The result is 42"

    def test_parse_final_answer_none(self):
        text = "Thought: not done yet\nAction: calculator"
        assert _parse_final_answer(text) is None

    def test_parse_action(self):
        text = "Thought: I'll use echo\nAction: echo\nAction Input: hello"
        action, action_input = _parse_action(text)
        assert action == "echo"
        assert action_input == "hello"

    def test_parse_thought(self):
        text = "Thought: I should calculate this\nAction: calc"
        thought = _parse_thought(text)
        assert "calculate" in thought

    def test_parse_multiline_final_answer(self):
        text = "Thought: done\nFinal Answer: Line one\nLine two"
        result = _parse_final_answer(text)
        assert "Line one" in result


# ------------------------------------------------------------------ #
# ReActAgent
# ------------------------------------------------------------------ #


class TestReActAgent:
    @pytest.mark.asyncio
    async def test_final_answer_first_step(self):
        llm = make_mock_llm(["Thought: I know the answer.\nFinal Answer: Paris"])
        agent = ReActAgent(llm=llm, tools=[])
        result = await agent.run("What is the capital of France?")
        assert result == "Paris"

    @pytest.mark.asyncio
    async def test_tool_call_then_final_answer(self):
        responses = [
            "Thought: I'll echo\nAction: echo\nAction Input: hello",
            "Thought: Got the result.\nFinal Answer: ECHO: hello",
        ]
        llm = make_mock_llm(responses)
        agent = ReActAgent(llm=llm, tools=[EchoTool()])
        result = await agent.run("Echo hello")
        assert result == "ECHO: hello"

    @pytest.mark.asyncio
    async def test_memory_records_steps(self):
        responses = [
            "Thought: echo time\nAction: echo\nAction Input: test",
            "Thought: done\nFinal Answer: ok",
        ]
        llm = make_mock_llm(responses)
        agent = ReActAgent(llm=llm, tools=[EchoTool()])
        await agent.run("test")
        assert len(agent.memory) == 1
        assert agent.memory.steps[0].action == "echo"

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_observation(self):
        responses = [
            "Thought: use unknown\nAction: unknown_tool\nAction Input: x",
            "Thought: got error\nFinal Answer: failed",
        ]
        llm = make_mock_llm(responses)
        agent = ReActAgent(llm=llm, tools=[])
        result = await agent.run("test")
        # Agent should handle gracefully and eventually get Final Answer
        assert result == "failed"
        assert "Error" in agent.memory.steps[0].observation

    @pytest.mark.asyncio
    async def test_tool_error_caught(self):
        responses = [
            "Thought: use bad\nAction: bad_tool\nAction Input: x",
            "Thought: handled error\nFinal Answer: recovered",
        ]
        llm = make_mock_llm(responses)
        agent = ReActAgent(llm=llm, tools=[ErrorTool()])
        result = await agent.run("test")
        assert result == "recovered"
        assert "Tool error" in agent.memory.steps[0].observation

    @pytest.mark.asyncio
    async def test_max_iterations_respected(self):
        # LLM never gives Final Answer — always uses tool
        always_action = "Thought: use echo\nAction: echo\nAction Input: x"
        llm = make_mock_llm([always_action] * 20)
        agent = ReActAgent(llm=llm, tools=[EchoTool()], max_iterations=3)
        result = await agent.run("test")
        assert "unable" in result.lower()
        assert len(agent.memory) == 3

    @pytest.mark.asyncio
    async def test_memory_cleared_on_new_run(self):
        llm = make_mock_llm(
            [
                "Thought: done\nFinal Answer: A",
                "Thought: done\nFinal Answer: B",
            ]
        )
        agent = ReActAgent(llm=llm, tools=[])
        await agent.run("first")
        assert len(agent.memory) == 0  # no tool calls
        await agent.run("second")
        assert len(agent.memory) == 0

    @pytest.mark.asyncio
    async def test_stream_yields_words(self):
        llm = make_mock_llm(["Final Answer: Hello world"])
        agent = ReActAgent(llm=llm, tools=[])
        tokens = []
        async for t in agent.stream("test"):
            tokens.append(t)
        assert len(tokens) > 0
        assert "Hello" in " ".join(tokens)

    @pytest.mark.asyncio
    async def test_no_format_falls_back_to_raw_response(self):
        llm = make_mock_llm(["The answer is simply 42."])
        agent = ReActAgent(llm=llm, tools=[])
        result = await agent.run("What is 42?")
        assert "42" in result
