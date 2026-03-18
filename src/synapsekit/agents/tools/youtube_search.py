"""YouTube Search Tool: search YouTube for videos."""

from __future__ import annotations

from typing import Any

from ..base import BaseTool, ToolResult


class YouTubeSearchTool(BaseTool):
    """Search YouTube for videos.

    Requires ``youtube-search-python``: ``pip install synapsekit[youtube]``

    Usage::

        tool = YouTubeSearchTool()
        result = await tool.run(query="python tutorial")
    """

    name = "youtube_search"
    description = (
        "Search YouTube for videos. "
        "Input: a search query. "
        "Returns: video titles, channels, durations, URLs, and view counts."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query for YouTube",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of videos to return (default: 5)",
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
            from youtubesearchpython import VideosSearch
        except ImportError:
            raise ImportError(
                "youtube-search-python package required: pip install synapsekit[youtube]"
            ) from None

        try:
            import asyncio

            loop = asyncio.get_event_loop()

            def _search():
                search = VideosSearch(search_query, limit=max_results)
                return search.result()

            response = await loop.run_in_executor(None, _search)

            videos = response.get("result", [])
            if not videos:
                return ToolResult(output="No results found.")

            results = []
            for i, video in enumerate(videos, 1):
                title = video.get("title", "Untitled")
                channel = video.get("channel", {}).get("name", "Unknown")
                duration = video.get("duration", "N/A")
                url = video.get("link", "")
                views = video.get("viewCount", {}).get("short", "N/A")
                results.append(
                    f"{i}. **{title}**\n"
                    f"   Channel: {channel}\n"
                    f"   Duration: {duration} | Views: {views}\n"
                    f"   URL: {url}"
                )

            return ToolResult(output="\n\n".join(results))
        except Exception as e:
            return ToolResult(output="", error=f"YouTube search failed: {e}")
