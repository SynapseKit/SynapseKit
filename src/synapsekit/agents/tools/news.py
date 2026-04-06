from __future__ import annotations

import json
import logging
import os
from typing import Any

from ..base import BaseTool, ToolResult

log = logging.getLogger(__name__)

_BASE_URL = "https://newsapi.org/v2"


class NewsTool(BaseTool):
    """Fetch news headlines and search articles via NewsAPI.

    Auth via constructor arg or ``NEWS_API_KEY`` env var.
    Uses stdlib ``urllib`` only.
    """

    name = "news"
    description = (
        "Fetch top headlines or search news articles. "
        "Actions: get_headlines, search."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "Action to perform",
                "enum": ["get_headlines", "search"],
            },
            "query": {
                "type": "string",
                "description": "Search query (required for search action)",
            },
            "country": {
                "type": "string",
                "description": "2-letter country code for headlines (default: us)",
            },
            "category": {
                "type": "string",
                "description": "News category for headlines",
                "enum": [
                    "business", "entertainment", "general",
                    "health", "science", "sports", "technology",
                ],
            },
            "from_date": {
                "type": "string",
                "description": "Oldest article date (ISO 8601, e.g. 2024-01-15)",
            },
            "sort_by": {
                "type": "string",
                "description": "Sort order for search results",
                "enum": ["relevancy", "popularity", "publishedAt"],
            },
        },
        "required": ["action"],
    }

    def __init__(self, api_key: str | None = None) -> None:
        self._key = api_key or os.environ.get("NEWS_API_KEY", "")

    async def run(self, action: str = "", **kwargs: Any) -> ToolResult:
        if not action:
            return ToolResult(output="", error="No action specified.")
        if not self._key:
            return ToolResult(output="", error="NEWS_API_KEY is required.")

        handlers = {
            "get_headlines": self._get_headlines,
            "search": self._search,
        }
        handler = handlers.get(action)
        if handler is None:
            return ToolResult(
                output="",
                error=f"Unknown action: {action}. Must be one of: {', '.join(handlers)}",
            )

        try:
            return await handler(**kwargs)
        except Exception as e:
            return ToolResult(output="", error=f"NewsAPI error: {e}")

    async def _get_headlines(self, country: str = "us", category: str = "", **_: Any) -> ToolResult:
        params = {"country": country, "apiKey": self._key}
        if category:
            params["category"] = category

        data = await self._get("/top-headlines", params)
        return self._format(data)

    async def _search(self, query: str = "", from_date: str = "", sort_by: str = "", **_: Any) -> ToolResult:
        if not query:
            return ToolResult(output="", error="query is required for search action.")

        params = {"q": query, "apiKey": self._key}
        if from_date:
            params["from"] = from_date
        if sort_by:
            params["sortBy"] = sort_by

        data = await self._get("/everything", params)
        return self._format(data)

    async def _get(self, endpoint: str, params: dict[str, str]) -> dict[str, Any]:
        import asyncio
        import urllib.parse
        import urllib.request

        url = f"{_BASE_URL}{endpoint}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "SynapseKit/1.0"})
        loop = asyncio.get_event_loop()

        def _fetch():
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())

        return await loop.run_in_executor(None, _fetch)

    def _format(self, data: dict[str, Any]) -> ToolResult:
        if data.get("status") != "ok":
            msg = data.get("message", "unknown error")
            return ToolResult(output="", error=f"API returned error: {msg}")

        articles = data.get("articles", [])
        if not articles:
            return ToolResult(output="No articles found.")

        results = []
        for a in articles:
            results.append({
                "title": a.get("title", ""),
                "source": a.get("source", {}).get("name", ""),
                "url": a.get("url", ""),
                "description": a.get("description", ""),
                "publishedAt": a.get("publishedAt", ""),
            })

        return ToolResult(output=json.dumps(results, indent=2))
