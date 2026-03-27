"""Bing Search Tool: web search via Bing Web Search API v7."""

from __future__ import annotations

import os
from typing import Any

from ..base import BaseTool, ToolResult


class BingSearchTool(BaseTool):
    """Web search via the Bing Web Search API v7.

    Uses the ``Ocp-Apim-Subscription-Key`` header for authentication.
    Requires ``httpx``: ``pip install httpx``

    Usage::

        tool = BingSearchTool(api_key="YOUR_KEY")
        result = await tool.run(query="latest AI news")
    """

    name = "bing_search"
    description = (
        "Search the web using Bing Web Search API. "
        "Input: a search query. "
        "Returns: relevant web results with titles, URLs, and descriptions."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query",
            },
            "count": {
                "type": "integer",
                "description": "Number of results to return (default: 5, max: 50)",
                "default": 5,
            },
            "market": {
                "type": "string",
                "description": "Market code (e.g., en-US, en-GB). Default: en-US",
                "default": "en-US",
            },
        },
        "required": ["query"],
    }

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("BING_API_KEY")

    async def run(
        self,
        query: str = "",
        count: int = 5,
        market: str = "en-US",
        **kwargs: Any,
    ) -> ToolResult:
        search_query = query or kwargs.get("input", "")
        if not search_query:
            return ToolResult(output="", error="No search query provided.")

        if not self._api_key:
            return ToolResult(output="", error="No BING_API_KEY configured.")

        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required: pip install httpx") from None

        url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {
            "Ocp-Apim-Subscription-Key": self._api_key,
        }
        params = {
            "q": search_query,
            "count": min(count, 50),
            "mkt": market,
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()

            web_pages = data.get("webPages", {})
            results_list = web_pages.get("value", [])

            if not results_list:
                return ToolResult(output="No results found.")

            results = []
            for i, r in enumerate(results_list, 1):
                title = r.get("name", "Untitled")
                result_url = r.get("url", "")
                snippet = r.get("snippet", "")[:300]
                results.append(f"{i}. **{title}**\n   URL: {result_url}\n   {snippet}")

            return ToolResult(output="\n\n".join(results))
        except httpx.HTTPStatusError as e:
            return ToolResult(output="", error=f"Bing Search API error: {e.response.status_code}")
        except Exception as e:
            return ToolResult(output="", error=f"Bing Search error: {e}")
