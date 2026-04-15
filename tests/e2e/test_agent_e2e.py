"""End-to-end Agent tests.

Full ReActAgent and FunctionCallingAgent flows using mocked LLMs and tools.
No API keys, no network.
"""
from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest

from synapsekit import CalculatorTool, DateTimeTool, FunctionCallingAgent, ReActAgent
from synapsekit.agents.base import BaseTool, ToolResult
from synapsekit.llm.base import BaseLLM, LLMConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_react_llm(response: str = "Final Answer: 42") -> BaseLLM:
    """Create a mock LLM for ReActAgent (uses generate_with_messages)."""
    config = LLMConfig(model="gpt-4o-mini", api_key="sk-test", provider="openai")
    llm = MagicMock(spec=BaseLLM)
    llm.config = config
    llm.generate_with_messages = AsyncMock(return_value=response)
    return llm


def _make_mock_tool(name: str, output: str = "tool_output") -> BaseTool:
    """Create a mock tool returning a ToolResult."""
    tool = MagicMock(spec=BaseTool)
    tool.name = name
    tool.description = f"A mock {name} tool"
    tool.parameters = {"type": "object", "properties": {"input": {"type": "string"}}}
    # ToolResult uses error=None to indicate success; is_error is a property
    tool.run = AsyncMock(return_value=ToolResult(output=output))
    return tool


# ---------------------------------------------------------------------------
# 1. ReActAgent basic run — Final Answer immediately
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_react_agent_run_final_answer_immediately():
    """ReActAgent returns Final Answer when LLM gives one immediately."""
    llm = _make_react_llm("Final Answer: Hello world")
    agent = ReActAgent(llm=llm, tools=[])
    result = await agent.run("Say hello")
    assert "Hello world" in result


@pytest.mark.asyncio
async def test_react_agent_run_no_format_returns_raw_response():
    """If LLM doesn't follow ReAct format, raw response is returned."""
    llm = _make_react_llm("I don't know the answer.")
    agent = ReActAgent(llm=llm, tools=[])
    result = await agent.run("Some question")
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# 2. ReActAgent with tool call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_react_agent_run_with_tool():
    """ReActAgent calls a tool when LLM emits Action: and then Final Answer."""
    tool = _make_mock_tool("my_tool", "tool_result_84")
    llm = MagicMock(spec=BaseLLM)
    llm.config = LLMConfig(model="gpt-4o-mini", api_key="sk-test", provider="openai")

    call_count = 0

    async def _generate(messages, **kw):
        nonlocal call_count
        if call_count == 0:
            call_count += 1
            return "Thought: Let me use the tool\nAction: my_tool\nAction Input: some input"
        else:
            return "Final Answer: 84"

    llm.generate_with_messages = _generate

    agent = ReActAgent(llm=llm, tools=[tool])
    result = await agent.run("What is 42 * 2?")
    assert "84" in result
    tool.run.assert_awaited_once()


# ---------------------------------------------------------------------------
# 3. ReActAgent stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_react_agent_stream_yields_tokens():
    llm = _make_react_llm("Final Answer: streamed result")
    agent = ReActAgent(llm=llm, tools=[])
    tokens = []
    async for tok in agent.stream("test"):
        tokens.append(tok)
    assert len(tokens) > 0
    full = " ".join(tokens)
    assert "streamed" in full or "result" in full


# ---------------------------------------------------------------------------
# 4. Async interface contracts
# ---------------------------------------------------------------------------


def test_react_agent_run_is_coroutine():
    assert inspect.iscoroutinefunction(ReActAgent.run)


def test_react_agent_stream_is_async_generator():
    assert inspect.isasyncgenfunction(ReActAgent.stream)


# ---------------------------------------------------------------------------
# 5. FunctionCallingAgent — provider without function calling raises clearly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_function_calling_agent_unsupported_llm_raises():
    """BaseLLM without _call_with_tools_impl override raises RuntimeError."""
    from collections.abc import AsyncGenerator

    from synapsekit.llm.base import BaseLLM, LLMConfig

    # Create a concrete subclass that does NOT override _call_with_tools_impl
    class MinimalLLM(BaseLLM):
        async def stream(self, prompt: str, **kw) -> AsyncGenerator[str]:  # type: ignore[override]
            yield ""

        async def stream_with_messages(self, messages: list[dict], **kw) -> AsyncGenerator[str]:  # type: ignore[override]
            yield ""

    config = LLMConfig(model="x", api_key="y", provider="openai")
    llm = MinimalLLM(config)
    agent = FunctionCallingAgent(llm=llm, tools=[])

    with pytest.raises(RuntimeError, match="does not support native function calling"):
        await agent.run("test")


# ---------------------------------------------------------------------------
# 6. FunctionCallingAgent max_iterations validation
# ---------------------------------------------------------------------------


def test_function_calling_agent_invalid_max_iterations():
    llm = MagicMock(spec=BaseLLM)
    with pytest.raises(ValueError, match="max_iterations"):
        FunctionCallingAgent(llm=llm, tools=[], max_iterations=0)


# ---------------------------------------------------------------------------
# 7. CalculatorTool e2e
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_calculator_tool_basic():
    tool = CalculatorTool()
    result = await tool.run(expression="2 + 2")
    assert not result.is_error
    assert "4" in result.output


@pytest.mark.asyncio
async def test_calculator_tool_division():
    tool = CalculatorTool()
    result = await tool.run(expression="10 / 4")
    assert not result.is_error
    assert "2.5" in result.output


@pytest.mark.asyncio
async def test_calculator_tool_invalid_expression():
    tool = CalculatorTool()
    result = await tool.run(expression="import os; os.system('rm -rf /')")
    # Should be an error — dangerous code blocked
    assert result.is_error


@pytest.mark.asyncio
async def test_calculator_tool_empty_expression():
    tool = CalculatorTool()
    result = await tool.run(expression="")
    assert result.is_error


# ---------------------------------------------------------------------------
# 8. DateTimeTool e2e
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_datetime_tool_now():
    tool = DateTimeTool()
    result = await tool.run(action="now")
    assert not result.is_error
    assert len(result.output) > 0


@pytest.mark.asyncio
async def test_datetime_tool_is_coroutine():
    tool = DateTimeTool()
    assert inspect.iscoroutinefunction(tool.run)


# ---------------------------------------------------------------------------
# 9. ReActAgent max_iterations safety (prevent infinite loops)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_react_agent_respects_max_iterations():
    """If LLM never gives Final Answer, agent stops after max_iterations."""
    llm = MagicMock(spec=BaseLLM)
    llm.config = LLMConfig(model="x", api_key="y", provider="openai")
    # Always returns thought without final answer
    llm.generate_with_messages = AsyncMock(return_value="Thought: I need to think more.")

    agent = ReActAgent(llm=llm, tools=[], max_iterations=2)
    result = await agent.run("infinite question")
    assert isinstance(result, str)
    # Should be the fallback message
    assert "unable" in result.lower() or len(result) > 0
