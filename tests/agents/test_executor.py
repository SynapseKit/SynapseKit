"""Tests for AgentExecutor and ToolRegistry."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from synapsekit.agents.executor import AgentConfig, AgentExecutor
from synapsekit.agents.registry import ToolRegistry
from synapsekit.agents.base import BaseTool, ToolResult
from synapsekit.agents.tools.calculator import CalculatorTool


# ------------------------------------------------------------------ #
# ToolRegistry
# ------------------------------------------------------------------ #

class TestToolRegistry:
    def test_get_existing(self):
        calc = CalculatorTool()
        reg = ToolRegistry([calc])
        assert reg.get("calculator") is calc

    def test_get_missing_raises(self):
        reg = ToolRegistry([])
        with pytest.raises(KeyError, match="calculator"):
            reg.get("calculator")

    def test_contains(self):
        reg = ToolRegistry([CalculatorTool()])
        assert "calculator" in reg
        assert "unknown" not in reg

    def test_schemas(self):
        reg = ToolRegistry([CalculatorTool()])
        schemas = reg.schemas()
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"

    def test_anthropic_schemas(self):
        reg = ToolRegistry([CalculatorTool()])
        schemas = reg.anthropic_schemas()
        assert "input_schema" in schemas[0]

    def test_describe(self):
        reg = ToolRegistry([CalculatorTool()])
        desc = reg.describe()
        assert "calculator" in desc

    def test_len(self):
        reg = ToolRegistry([CalculatorTool()])
        assert len(reg) == 1

    def test_iter(self):
        tools = [CalculatorTool()]
        reg = ToolRegistry(tools)
        listed = list(reg)
        assert len(listed) == 1


# ------------------------------------------------------------------ #
# AgentExecutor — react
# ------------------------------------------------------------------ #

class TestAgentExecutorReAct:
    def _make_react_llm(self, answer="42"):
        llm = MagicMock()
        llm.generate_with_messages = AsyncMock(
            return_value=f"Thought: I know.\nFinal Answer: {answer}"
        )
        return llm

    @pytest.mark.asyncio
    async def test_run_react(self):
        llm = self._make_react_llm("Paris")
        executor = AgentExecutor(AgentConfig(llm=llm, tools=[], agent_type="react"))
        result = await executor.run("Capital of France?")
        assert result == "Paris"

    def test_run_sync(self):
        llm = self._make_react_llm("sync answer")
        executor = AgentExecutor(AgentConfig(llm=llm, tools=[], agent_type="react"))
        result = executor.run_sync("test?")
        assert result == "sync answer"

    @pytest.mark.asyncio
    async def test_stream_react(self):
        llm = self._make_react_llm("hello world")
        executor = AgentExecutor(AgentConfig(llm=llm, tools=[], agent_type="react"))
        tokens = []
        async for t in executor.stream("test?"):
            tokens.append(t)
        assert len(tokens) > 0

    @pytest.mark.asyncio
    async def test_memory_accessible(self):
        llm = self._make_react_llm("done")
        executor = AgentExecutor(AgentConfig(llm=llm, tools=[], agent_type="react"))
        await executor.run("test")
        assert executor.memory is not None


# ------------------------------------------------------------------ #
# AgentExecutor — function_calling
# ------------------------------------------------------------------ #

class TestAgentExecutorFunctionCalling:
    def _make_fc_llm(self, answer="done"):
        llm = MagicMock()
        llm.call_with_tools = AsyncMock(
            return_value={"content": answer, "tool_calls": None}
        )
        return llm

    @pytest.mark.asyncio
    async def test_run_function_calling(self):
        llm = self._make_fc_llm("function answer")
        executor = AgentExecutor(AgentConfig(
            llm=llm, tools=[], agent_type="function_calling"
        ))
        result = await executor.run("test?")
        assert result == "function answer"

    def test_unknown_agent_type_raises(self):
        llm = MagicMock()
        with pytest.raises(ValueError, match="Unknown agent_type"):
            AgentExecutor(AgentConfig(llm=llm, tools=[], agent_type="invalid"))

    @pytest.mark.asyncio
    async def test_with_calculator_tool(self):
        llm = MagicMock()
        llm.call_with_tools = AsyncMock(side_effect=[
            {
                "content": None,
                "tool_calls": [{"id": "t1", "name": "calculator", "arguments": {"expression": "2 ** 8"}}],
            },
            {"content": "The answer is 256.", "tool_calls": None},
        ])
        executor = AgentExecutor(AgentConfig(
            llm=llm,
            tools=[CalculatorTool()],
            agent_type="function_calling",
        ))
        result = await executor.run("What is 2 to the power of 8?")
        assert result == "The answer is 256."
        assert executor.memory.steps[0].observation == "256"
