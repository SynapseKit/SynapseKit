from __future__ import annotations

import asyncio

from .base import Document


class RSSLoader:
    """Load articles from RSS or Atom feeds."""

    def __init__(self, url: str) -> None:
        self._url = url

    def load(self) -> list[Document]:
        try:
            import feedparser
        except ImportError:
            raise ImportError("feedparser required: pip install synapsekit[rss]") from None

        feed = feedparser.parse(self._url)
        documents = []

        for entry in feed.entries:
            text = entry.get("content", [{"value": entry.get("summary", "")}])[0].get(
                "value", entry.get("summary", "")
            )

            metadata = {
                "title": entry.get("title", ""),
                "published": entry.get("published", ""),
                "link": entry.get("link", ""),
                "author": entry.get("author", ""),
            }

            metadata = {k: v for k, v in metadata.items() if v}

            documents.append(Document(text=text, metadata=metadata))

        return documents

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)
