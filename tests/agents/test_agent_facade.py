import pytest

from synapsekit import agent, tool


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 22°C in {city}"


@pytest.mark.asyncio
async def test_simple_agent_async():
    # Use a cheap fast model or mock if possible, but for integration we can use a known provider.
    # To avoid API calls in CI without keys, we can use a mock LLM. But since synapsekit provides no built-in MockLLM out of the box in the test setup, we will just construct the agent to ensure factory works.

    my_agent = agent(
        model="gpt-4o-mini",
        api_key="dummy",
        tools=[get_weather],
    )

    assert my_agent._executor is not None
    assert my_agent._memory is None
    assert len(my_agent._executor.config.tools) == 1
    assert my_agent._executor.config.tools[0].name == "get_weather"


def test_simple_agent_sync():
    my_agent = agent(
        model="gpt-4o-mini",
        api_key="dummy",
        tools=[get_weather],
        memory=True,
    )

    assert my_agent._memory is not None
    assert len(my_agent._memory) == 0


def test_tool_decorator_no_parens():
    @tool
    def multiply(a: int, b: int) -> str:
        """Multiply two numbers."""
        return str(a * b)

    assert multiply.name == "multiply"
    assert multiply.description == "Multiply two numbers."
    assert multiply.parameters["type"] == "object"
    assert "a" in multiply.parameters["properties"]
    assert "b" in multiply.parameters["properties"]
