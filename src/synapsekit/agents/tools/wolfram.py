"""Wolfram Alpha Tool: computational answers via Wolfram Alpha API."""

from __future__ import annotations

import os
from typing import Any

from ..base import BaseTool, ToolResult


class WolframAlphaTool(BaseTool):
    """Computational answers via Wolfram Alpha API.

    Requires ``wolframalpha``: ``pip install synapsekit[wolfram]``

    Usage::

        tool = WolframAlphaTool(api_key="YOUR_APP_ID")
        result = await tool.run(query="integrate x^2")
    """

    name = "wolfram_alpha"
    description = (
        "Query Wolfram Alpha for math, science, and factual computation. "
        "Input: a query string. "
        "Returns: the short answer text."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The Wolfram Alpha query",
            },
        },
        "required": ["query"],
    }

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("WOLFRAM_API_KEY")

    async def run(self, query: str = "", **kwargs: Any) -> ToolResult:
        search_query = query or kwargs.get("input", "")
        if not search_query:
            return ToolResult(output="", error="No query provided.")

        if not self._api_key:
            return ToolResult(output="", error="No WOLFRAM_API_KEY configured.")

        try:
            import wolframalpha
        except ImportError:
            raise ImportError(
                "wolframalpha package required: pip install synapsekit[wolfram]"
            ) from None

        try:
            import asyncio

            loop = asyncio.get_event_loop()

            def _query():
                client = wolframalpha.Client(self._api_key)
                return client.query(search_query)

            response = await loop.run_in_executor(None, _query)

            result_text = _extract_short_answer(response)
            if not result_text:
                return ToolResult(output="No results found.")

            return ToolResult(output=result_text)
        except Exception as e:
            return ToolResult(output="", error=f"Wolfram Alpha error: {e}")


def _extract_short_answer(response: Any) -> str | None:
    results = getattr(response, "results", None)
    if results is not None:
        for result in results:
            text = getattr(result, "text", None)
            if isinstance(text, str) and text:
                return text

    pods = getattr(response, "pods", None)
    if isinstance(pods, (list, tuple)):
        for pod in pods:
            subpods = getattr(pod, "subpods", None)
            if not subpods:
                continue
            for subpod in subpods:
                text = getattr(subpod, "text", None)
                if isinstance(text, str) and text:
                    return text

    return None
