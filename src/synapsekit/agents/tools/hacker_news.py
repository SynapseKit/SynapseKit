from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from typing import Any

from ..base import BaseTool, ToolResult


class HackerNewsTool(BaseTool):
    """Fetch top stories or search HackerNews."""

    name = "hacker_news"
    description = (
        "Fetch top stories from Hacker News or get details about a specific item. "
        "Useful for catching up on tech news, discussions, and trending topics. "
        "Input: action 'top_stories' to fetch top stories, or 'get_item' with an 'item_id'."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["top_stories", "get_item"],
                "description": "The action to perform. 'top_stories' fetches the current top stories. 'get_item' fetches details of a specific item.",
            },
            "item_id": {
                "type": "integer",
                "description": "The ID of the item to fetch. Required if action is 'get_item'.",
            },
        },
        "required": ["action"],
    }

    async def run(
        self, action: str = "top_stories", item_id: int | None = None, **kwargs: Any
    ) -> ToolResult:
        """Run the tool."""
        # Fallback to kwargs if arguments were mapped directly
        _action = (
            kwargs.get("input", action) if action == "top_stories" and "input" in kwargs else action
        )
        if _action not in ("top_stories", "get_item"):
            _action = "top_stories"

        return await asyncio.to_thread(self._run_sync, _action, item_id)

    def _run_sync(self, action: str, item_id: int | None) -> ToolResult:
        if action == "top_stories":
            url = "https://hacker-news.firebaseio.com/v0/topstories.json"
            try:
                with urllib.request.urlopen(url, timeout=10) as response:
                    data = json.loads(response.read().decode("utf-8"))

                top_ids = data[:10]
                stories = []
                for story_id in top_ids:
                    story_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
                    with urllib.request.urlopen(story_url, timeout=5) as s_response:
                        story_data = json.loads(s_response.read().decode("utf-8"))
                        if story_data:
                            title = story_data.get("title", "")
                            url = story_data.get(
                                "url", f"https://news.ycombinator.com/item?id={story_id}"
                            )
                            score = story_data.get("score", 0)
                            stories.append(f"- {title} (Score: {score}) - {url}")

                output = "Top 10 Hacker News Stories:\n" + "\n".join(stories)
                return ToolResult(output=output)
            except Exception as e:
                return ToolResult(output="", error=f"Failed to fetch top stories: {e}")

        elif action == "get_item":
            if not item_id:
                return ToolResult(output="", error="item_id is required for get_item action.")
            url = f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json"
            try:
                with urllib.request.urlopen(url, timeout=10) as response:
                    data = json.loads(response.read().decode("utf-8"))
                    if not data:
                        return ToolResult(output="", error="Item not found.")
                    output = json.dumps(data, indent=2)
                    return ToolResult(output=output)
            except Exception as e:
                return ToolResult(output="", error=f"Failed to fetch item: {e}")
        else:
            return ToolResult(output="", error=f"Unknown action: {action}")
