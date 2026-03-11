from __future__ import annotations

from typing import Any

from ..base import BaseTool, ToolResult


class WebSearchTool(BaseTool):
    """Search the web using DuckDuckGo (no API key required)."""

    name = "web_search"
    description = (
        "Search the web for current information. "
        "Input: a search query string. "
        "Returns: a list of result titles, URLs, and snippets."
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
        },
        "required": ["query"],
    }

    async def run(self, query: str = "", max_results: int = 5, **kwargs: Any) -> ToolResult:
        search_query = query or kwargs.get("input", "")
        if not search_query:
            return ToolResult(output="", error="No search query provided.")

        try:
            from duckduckgo_search import DDGS
        except ImportError:
            raise ImportError(
                "duckduckgo-search required: pip install synapsekit[search]"
            ) from None

        try:
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(search_query, max_results=max_results):
                    title = r.get("title", "")
                    href = r.get("href", "")
                    body = r.get("body", "")
                    results.append(f"**{title}**\n{href}\n{body}")

            if not results:
                return ToolResult(output="No results found.")

            return ToolResult(output="\n\n".join(results))
        except Exception as e:
            return ToolResult(output="", error=f"Search failed: {e}")
