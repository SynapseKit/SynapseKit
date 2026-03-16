"""Tavily Search Tool: AI-optimized web search."""

from __future__ import annotations

import os
from typing import Any

from ..base import BaseTool, ToolResult


class TavilySearchTool(BaseTool):
    """AI-optimized web search via the Tavily API.

    Requires ``tavily-python``: ``pip install synapsekit[tavily]``

    Usage::

        tool = TavilySearchTool(api_key="tvly-...")
        result = await tool.run(query="latest AI breakthroughs")
    """

    name = "tavily_search"
    description = (
        "Search the web using Tavily's AI-optimized search engine. "
        "Input: a search query. "
        "Returns: relevant web results with titles, URLs, and content snippets."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 5)",
                "default": 5,
            },
            "search_depth": {
                "type": "string",
                "description": "Search depth: 'basic' or 'advanced' (default: 'basic')",
                "default": "basic",
            },
        },
        "required": ["query"],
    }

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("TAVILY_API_KEY")

    async def run(
        self,
        query: str = "",
        max_results: int = 5,
        search_depth: str = "basic",
        **kwargs: Any,
    ) -> ToolResult:
        search_query = query or kwargs.get("input", "")
        if not search_query:
            return ToolResult(output="", error="No search query provided.")

        try:
            from tavily import TavilyClient
        except ImportError:
            raise ImportError(
                "tavily-python package required: pip install synapsekit[tavily]"
            ) from None

        api_key = self._api_key
        if not api_key:
            return ToolResult(output="", error="No Tavily API key provided.")

        try:
            import asyncio

            loop = asyncio.get_event_loop()

            def _search():
                client = TavilyClient(api_key=api_key)
                return client.search(
                    query=search_query,
                    max_results=max_results,
                    search_depth=search_depth,
                )

            response = await loop.run_in_executor(None, _search)

            results_list = response.get("results", [])
            if not results_list:
                return ToolResult(output="No results found.")

            results = []
            for i, r in enumerate(results_list, 1):
                title = r.get("title", "Untitled")
                url = r.get("url", "")
                content = r.get("content", "")[:500]
                results.append(f"{i}. **{title}**\n   URL: {url}\n   {content}")

            return ToolResult(output="\n\n".join(results))
        except Exception as e:
            return ToolResult(output="", error=f"Tavily search failed: {e}")
