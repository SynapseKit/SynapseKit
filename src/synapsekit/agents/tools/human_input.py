"""Human Input Tool: allows agents to ask the user a question mid-execution."""

from __future__ import annotations

import asyncio
from typing import Any

from ..base import BaseTool, ToolResult


class HumanInputTool(BaseTool):
    """Tool that asks the user for input during agent execution.

    This enables human-in-the-loop agent workflows where the agent can
    request clarification or additional information from the user.

    Usage::

        tool = HumanInputTool()
        # Or with a custom input function:
        tool = HumanInputTool(input_fn=my_custom_input)

    By default, uses Python's built-in ``input()`` function.
    Pass a custom ``input_fn`` for non-terminal environments (web, API, etc.).
    """

    name = "human_input"
    description = (
        "Ask the user a question and get their response. "
        "Use this when you need clarification, additional information, "
        "or confirmation from the user before proceeding."
    )
    parameters = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask the user",
            },
        },
        "required": ["question"],
    }

    def __init__(self, input_fn: Any = None) -> None:
        self._input_fn = input_fn

    async def run(self, question: str = "", **kwargs: Any) -> ToolResult:
        prompt = question or kwargs.get("input", "")
        if not prompt:
            return ToolResult(output="", error="No question provided.")

        try:
            if self._input_fn is not None:
                result = self._input_fn(prompt)
                if asyncio.iscoroutine(result):
                    result = await result
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, input, f"\n{prompt}\n> ")

            return ToolResult(output=str(result))
        except Exception as e:
            return ToolResult(output="", error=f"Failed to get input: {e}")
