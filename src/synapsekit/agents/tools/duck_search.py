from __future__ import annotations

from typing import Any

from ..base import BaseTool, ToolResult


class DuckDuckGoSearchTool(BaseTool):
    """Search the web using DuckDuckGo with support for text and news searches."""

    name = "duckduckgo_search"
    description = (
        "Search the web using DuckDuckGo. "
        "Supports text and news search types. "
        "Returns a numbered list of results with title, URL, and snippet."
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
            "search_type": {
                "type": "string",
                "description": "Search type: 'text' (default) or 'news'",
                "enum": ["text", "news"],
                "default": "text",
            },
        },
        "required": ["query"],
    }

    async def run(
        self,
        query: str = "",
        max_results: int = 5,
        search_type: str = "text",
        **kwargs: Any,
    ) -> ToolResult:
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
                if search_type == "news":
                    raw = ddgs.news(search_query, max_results=max_results)
                else:
                    raw = ddgs.text(search_query, max_results=max_results)

                for i, r in enumerate(raw, 1):
                    title = r.get("title", "")
                    url = r.get("href", "") or r.get("url", "")
                    body = r.get("body", "") or r.get("excerpt", "")
                    results.append(f"{i}. {title}\n   {url}\n   {body}")

            if not results:
                return ToolResult(output="No results found.")

            return ToolResult(output="\n\n".join(results))
        except Exception as e:
            return ToolResult(output="", error=f"DuckDuckGo search failed: {e}")
