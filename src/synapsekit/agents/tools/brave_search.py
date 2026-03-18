"""Brave Search Tool: web search via the Brave Search API."""

from __future__ import annotations

import os
from typing import Any

from ..base import BaseTool, ToolResult


class BraveSearchTool(BaseTool):
    """Web search via the Brave Search API.

    Uses the ``X-Subscription-Token`` header for authentication.
    Stdlib ``urllib`` only — no extra dependencies.

    Usage::

        tool = BraveSearchTool(api_key="BSA...")
        result = await tool.run(query="latest AI news")
    """

    name = "brave_search"
    description = (
        "Search the web using Brave Search API. "
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
                "description": "Number of results to return (default: 5, max: 20)",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("BRAVE_API_KEY")

    async def run(self, query: str = "", count: int = 5, **kwargs: Any) -> ToolResult:
        search_query = query or kwargs.get("input", "")
        if not search_query:
            return ToolResult(output="", error="No search query provided.")

        if not self._api_key:
            return ToolResult(output="", error="No BRAVE_API_KEY configured.")

        try:
            import asyncio
            import json
            import urllib.request
            from urllib.parse import quote_plus

            url = (
                f"https://api.search.brave.com/res/v1/web/search"
                f"?q={quote_plus(search_query)}&count={min(count, 20)}"
            )
            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": self._api_key,
            }
            req = urllib.request.Request(url, headers=headers)

            loop = asyncio.get_event_loop()

            def _fetch():
                with urllib.request.urlopen(req, timeout=15) as resp:
                    return json.loads(resp.read().decode())

            data = await loop.run_in_executor(None, _fetch)

            web_results = data.get("web", {}).get("results", [])
            if not web_results:
                return ToolResult(output="No results found.")

            results = []
            for i, r in enumerate(web_results, 1):
                title = r.get("title", "Untitled")
                result_url = r.get("url", "")
                desc = r.get("description", "")[:300]
                results.append(f"{i}. **{title}**\n   URL: {result_url}\n   {desc}")

            return ToolResult(output="\n\n".join(results))
        except Exception as e:
            return ToolResult(output="", error=f"Brave Search error: {e}")
