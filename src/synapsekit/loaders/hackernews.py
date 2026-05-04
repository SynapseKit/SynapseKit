from __future__ import annotations

import asyncio
import json
import urllib.request
from typing import Any

from .base import Document

_VALID_STORY_TYPES = {"top", "new", "best", "ask", "show"}
_HN_BASE = "https://hacker-news.firebaseio.com/v0"


class HackerNewsLoader:
    """Load Hacker News stories using the official HN Firebase REST API.

    No authentication required; no extra dependencies (uses urllib).

    Example::

        loader = HackerNewsLoader(story_type="top", limit=10)
        docs = loader.load()
    """

    def __init__(
        self,
        story_type: str = "top",
        limit: int = 10,
    ) -> None:
        if story_type not in _VALID_STORY_TYPES:
            raise ValueError(
                f"story_type must be one of {_VALID_STORY_TYPES!r}, got {story_type!r}"
            )
        if limit <= 0:
            raise ValueError("limit must be greater than 0")

        self._story_type = story_type
        self._limit = limit

    def load(self) -> list[Document]:
        story_ids = self._fetch_story_ids()
        story_ids = story_ids[: self._limit]

        docs: list[Document] = []
        for item_id in story_ids:
            item = self._fetch_item(item_id)
            if not item:
                continue
            title = item.get("title", "")
            body = item.get("text") or item.get("url") or ""
            text = title + ("\n" + body if body else "")
            metadata: dict[str, Any] = {
                "source": "hackernews",
                "item_id": item_id,
                "score": item.get("score"),
                "by": item.get("by"),
                "time": item.get("time"),
                "url": item.get("url"),
                "story_type": self._story_type,
            }
            docs.append(Document(text=text, metadata=metadata))

        return docs

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)

    def _fetch_story_ids(self) -> list[int]:
        url = f"{_HN_BASE}/{self._story_type}stories.json"
        with urllib.request.urlopen(url) as resp:
            return json.loads(resp.read().decode())

    def _fetch_item(self, item_id: int) -> dict[str, Any] | None:
        url = f"{_HN_BASE}/item/{item_id}.json"
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read().decode())
        return data
