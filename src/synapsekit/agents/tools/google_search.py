from __future__ import annotations

import os
from typing import Any

from ..base import BaseTool, ToolResult


class GoogleSearchTool(BaseTool):
    """Search the web using Google Search via SerpAPI."""

    name = "google_search"
    description = (
        "Search Google for current information. "
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

    def __init__(self, api_key: str | None = None) -> None:
        """
        Args:
            api_key: SerpAPI key. If None, checks SERPAPI_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("SERPAPI_API_KEY")

    async def run(self, query: str = "", max_results: int = 5, **kwargs: Any) -> ToolResult:
        search_query = query or kwargs.get("input", "")
        if not search_query:
            return ToolResult(output="", error="No search query provided.")

        if not self.api_key:
            return ToolResult(
                output="",
                error="SerpAPI key not provided. Pass it to the constructor or set SERPAPI_API_KEY.",
            )

        try:
            from serpapi import GoogleSearch
        except ImportError:
            raise ImportError(
                "google-search-results required: pip install synapsekit[google-search]"
            ) from None

        try:
            params = {
                "q": search_query,
                "api_key": self.api_key,
                "num": max_results,
            }
            search = GoogleSearch(params)
            results = search.get_dict()

            if "error" in results:
                return ToolResult(output="", error=f"SerpAPI Error: {results['error']}")

            organic_results = results.get("organic_results", [])
            if not organic_results:
                return ToolResult(output="No results found.")

            formatted_results = []
            for r in organic_results[:max_results]:
                title = r.get("title", "")
                link = r.get("link", "")
                snippet = r.get("snippet", "")
                formatted_results.append(f"**{title}**\n{link}\n{snippet}")

            return ToolResult(output="\n\n".join(formatted_results))
        except Exception as e:
            return ToolResult(output="", error=f"Google Search failed: {e}")
