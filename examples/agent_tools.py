"""
ReAct Agent with Custom Tools
==============================

This example shows how to create a ReAct agent with built-in and custom tools.
The agent can reason about which tool to use and execute them step-by-step.

Prerequisites:
    pip install synapsekit[openai]

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/agent_tools.py
"""

import asyncio
import os
from datetime import datetime

from synapsekit import ReActAgent, tool, CalculatorTool, DateTimeTool


# Define a custom tool
@tool
def get_weather(location: str) -> str:
    """
    Get current weather for a location.
    
    Args:
        location: City name or location
        
    Returns:
        Weather description
    """
    # Mock weather data
    return f"The weather in {location} is sunny with 22°C temperature."


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """
    Convert currency from one type to another.
    
    Args:
        amount: Amount to convert
        from_currency: Source currency code (e.g., USD, EUR)
        to_currency: Target currency code
        
    Returns:
        Converted amount as string
    """
    # Mock conversion (in reality, fetch real rates)
    rates = {"USD": 1.0, "EUR": 0.85, "GBP": 0.73, "JPY": 110.0}
    if from_currency not in rates or to_currency not in rates:
        return "Currency not supported"
    
    result = amount * (rates[to_currency] / rates[from_currency])
    return f"{amount} {from_currency} = {result:.2f} {to_currency}"


async def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    # Create agent with multiple tools
    agent = ReActAgent(
        model="gpt-4o-mini",
        api_key=api_key,
        tools=[
            CalculatorTool(),
            DateTimeTool(),
            get_weather,
            convert_currency,
        ],
        verbose=True,
    )

    # Example queries that require different tools
    queries = [
        "What's the weather in Tokyo and how much is 100 USD in JPY?",
        "Calculate 15% tip on a $45.50 bill and tell me today's date.",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        result = await agent.run(query)
        print(f"\nFinal Answer: {result}")


if __name__ == "__main__":
    asyncio.run(main())
