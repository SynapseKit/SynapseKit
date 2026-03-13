"""Wikipedia Tool: search and fetch Wikipedia articles."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote_plus

from ..base import BaseTool, ToolResult


class WikipediaTool(BaseTool):
    """Search and fetch Wikipedia article summaries.

    Uses the Wikipedia REST API — no API key required, no extra dependencies.

    Usage::

        tool = WikipediaTool()
        result = await tool.run(query="Python programming language")
    """

    name = "wikipedia"
    description = (
        "Search Wikipedia and return article summaries. "
        "Input: a search query. "
        "Returns: the title and summary of the most relevant Wikipedia article."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query for Wikipedia",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of articles to return (default: 1)",
                "default": 1,
            },
        },
        "required": ["query"],
    }

    def __init__(self, max_chars: int = 4000) -> None:
        self._max_chars = max_chars

    async def run(self, query: str = "", max_results: int = 1, **kwargs: Any) -> ToolResult:
        search_query = query or kwargs.get("input", "")
        if not search_query:
            return ToolResult(output="", error="No search query provided.")

        try:
            import urllib.request

            # Search for articles
            search_url = (
                f"https://en.wikipedia.org/w/api.php?action=opensearch"
                f"&search={quote_plus(search_query)}&limit={max_results}&format=json"
            )

            import asyncio
            import json

            loop = asyncio.get_event_loop()

            def _fetch_search():
                req = urllib.request.Request(search_url, headers={"User-Agent": "SynapseKit/1.0"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    return json.loads(resp.read().decode())

            search_data = await loop.run_in_executor(None, _fetch_search)

            if len(search_data) < 2 or not search_data[1]:
                return ToolResult(output="No Wikipedia articles found.")

            results = []
            titles = search_data[1][:max_results]

            for title in titles:
                # Fetch article summary
                summary_url = (
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote_plus(title)}"
                )

                def _fetch_summary(url=summary_url):
                    req = urllib.request.Request(url, headers={"User-Agent": "SynapseKit/1.0"})
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        return json.loads(resp.read().decode())

                summary_data = await loop.run_in_executor(None, _fetch_summary)

                article_title = summary_data.get("title", title)
                extract = summary_data.get("extract", "No summary available.")
                url = summary_data.get("content_urls", {}).get("desktop", {}).get("page", "")

                if len(extract) > self._max_chars:
                    extract = extract[: self._max_chars] + "..."

                results.append(f"**{article_title}**\n{url}\n\n{extract}")

            return ToolResult(output="\n\n---\n\n".join(results))
        except Exception as e:
            return ToolResult(output="", error=f"Wikipedia search failed: {e}")
