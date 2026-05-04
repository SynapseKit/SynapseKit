"""Tests for SimpleAgent facade and agent() factory (issue #599)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from synapsekit import agent, tool

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 22\u00b0C in {city}"


def _make_react_llm(answer: str = "42") -> MagicMock:
    llm = MagicMock()
    llm.generate_with_messages = AsyncMock(return_value=f"Thought: I know.\nFinal Answer: {answer}")
    return llm


# ------------------------------------------------------------------ #
# Construction / factory
# ------------------------------------------------------------------ #


def test_agent_factory_creates_simple_agent():
    """agent() returns a SimpleAgent with a wired-up executor."""
    my_agent = agent(
        model="gpt-4o-mini",
        api_key="dummy",
        tools=[get_weather],
    )

    assert my_agent._executor is not None
    assert my_agent._memory is None
    assert len(my_agent._executor.config.tools) == 1
    assert my_agent._executor.config.tools[0].name == "get_weather"


def test_agent_factory_memory_flag():
    """memory=True wires a ConversationMemory instance."""
    my_agent = agent(
        model="gpt-4o-mini",
        api_key="dummy",
        memory=True,
    )

    assert my_agent._memory is not None
    assert len(my_agent._memory) == 0


def test_default_config_values():
    """Factory sets sensible defaults: react loop, 10 max iterations, helpful prompt."""
    my_agent = agent(model="gpt-4o-mini", api_key="dummy")
    cfg = my_agent._executor.config
    assert cfg.agent_type == "react"
    assert cfg.max_iterations == 10
    assert cfg.system_prompt == "You are a helpful AI assistant."


# ------------------------------------------------------------------ #
# Provider auto-detection
# ------------------------------------------------------------------ #


def test_provider_autodetect_openai():
    """gpt-* model names auto-select the openai provider."""
    my_agent = agent(model="gpt-4o-mini", api_key="dummy")
    assert my_agent._executor.config.llm.config.provider == "openai"


def test_provider_autodetect_anthropic():
    """claude-* model names auto-select the anthropic provider."""
    my_agent = agent(model="claude-3-haiku-20240307", api_key="dummy")
    assert my_agent._executor.config.llm.config.provider == "anthropic"


def test_provider_autodetect_gemini():
    """gemini-* model names auto-select the gemini provider."""
    my_agent = agent(model="gemini-1.5-flash", api_key="dummy")
    assert my_agent._executor.config.llm.config.provider == "gemini"


def test_provider_explicit_override():
    """Explicit provider kwarg overrides model-name auto-detection."""
    my_agent = agent(model="llama-3-8b-8192", api_key="dummy", provider="groq")
    assert my_agent._executor.config.llm.config.provider == "groq"


# ------------------------------------------------------------------ #
# @tool decorator
# ------------------------------------------------------------------ #


def test_tool_decorator_no_parens():
    """@tool without parens infers name and description from the function."""

    @tool
    def multiply(a: int, b: int) -> str:
        """Multiply two numbers."""
        return str(a * b)

    assert multiply.name == "multiply"
    assert multiply.description == "Multiply two numbers."
    assert multiply.parameters["type"] == "object"
    assert "a" in multiply.parameters["properties"]
    assert "b" in multiply.parameters["properties"]


def test_tool_decorator_with_parens():
    """@tool(name=..., description=...) allows explicit metadata."""

    @tool(name="add_nums", description="Add two numbers together.")
    def add(a: int, b: int) -> str:
        return str(a + b)

    assert add.name == "add_nums"
    assert add.description == "Add two numbers together."


# ------------------------------------------------------------------ #
# SimpleAgent.arun (async)
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_arun_returns_final_answer():
    """SimpleAgent.arun() extracts the Final Answer from the ReAct loop."""
    from synapsekit.agents.executor import AgentConfig, AgentExecutor
    from synapsekit.agents.facade import SimpleAgent

    llm = _make_react_llm("Sunny, 22\u00b0C in Tokyo")
    executor = AgentExecutor(AgentConfig(llm=llm, tools=[], agent_type="react"))
    my_agent = SimpleAgent(executor=executor)

    result = await my_agent.arun("What's the weather in Tokyo?")
    assert result == "Sunny, 22\u00b0C in Tokyo"


@pytest.mark.asyncio
async def test_arun_with_memory_flag():
    """agent() with memory=True accumulates turns via arun()."""
    from synapsekit.agents.executor import AgentConfig, AgentExecutor
    from synapsekit.agents.facade import SimpleAgent
    from synapsekit.memory.conversation import ConversationMemory

    llm = MagicMock()
    llm.generate_with_messages = AsyncMock(
        side_effect=[
            "Thought: ok.\nFinal Answer: Paris",
            "Thought: ok.\nFinal Answer: Eiffel Tower",
        ]
    )
    executor = AgentExecutor(AgentConfig(llm=llm, tools=[], agent_type="react"))
    conv_memory = ConversationMemory()
    my_agent = SimpleAgent(executor=executor, memory=conv_memory)

    await my_agent.arun("Capital of France?")
    assert len(my_agent._memory) == 2  # user + assistant

    await my_agent.arun("Famous landmark there?")
    assert len(my_agent._memory) == 4  # second turn appended


# ------------------------------------------------------------------ #
# SimpleAgent.run (sync)
# ------------------------------------------------------------------ #


def test_run_returns_final_answer():
    """SimpleAgent.run() is the sync wrapper that returns the parsed answer."""
    from synapsekit.agents.executor import AgentConfig, AgentExecutor
    from synapsekit.agents.facade import SimpleAgent

    llm = _make_react_llm("42")
    executor = AgentExecutor(AgentConfig(llm=llm, tools=[], agent_type="react"))
    my_agent = SimpleAgent(executor=executor)

    result = my_agent.run("What is 6 x 7?")
    assert result == "42"


def test_run_memory_accumulates():
    """memory=True: conversation history grows across multiple run() calls."""
    from synapsekit.agents.executor import AgentConfig, AgentExecutor
    from synapsekit.agents.facade import SimpleAgent
    from synapsekit.memory.conversation import ConversationMemory

    llm = MagicMock()
    llm.generate_with_messages = AsyncMock(
        side_effect=[
            "Thought: ok.\nFinal Answer: Paris",
            "Thought: ok.\nFinal Answer: Eiffel Tower",
        ]
    )
    executor = AgentExecutor(AgentConfig(llm=llm, tools=[], agent_type="react"))
    conv_memory = ConversationMemory()
    my_agent = SimpleAgent(executor=executor, memory=conv_memory)

    my_agent.run("What is the capital of France?")
    assert len(my_agent._memory) == 2

    my_agent.run("What is the famous landmark there?")
    assert len(my_agent._memory) == 4
