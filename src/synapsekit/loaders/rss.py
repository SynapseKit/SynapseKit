from __future__ import annotations

import asyncio
from urllib.parse import urlparse

from ._url_guard import validate_public_url
from .base import Document


class RSSLoader:
    """Load articles from RSS or Atom feeds."""

    def __init__(self, url: str) -> None:
        self._url = url

    def _validate(self) -> None:
        """SSRF guard for remote feeds, without breaking local-file usage.

        ``feedparser.parse`` happily accepts ``file://`` URLs (local file read)
        and can be pointed at internal HTTP services. For ``http(s)`` URLs we
        run the shared fail-closed SSRF guard; any other explicit scheme
        (``file``, ``ftp``, ...) is rejected. A bare path or inline XML string
        (no scheme) is still allowed so existing local usage keeps working.
        """
        scheme = urlparse(self._url).scheme.lower()
        if len(scheme) == 1:
            scheme = ""
            
        if scheme in ("http", "https"):
            validate_public_url(self._url)
        elif scheme:
            raise ValueError(f"RSS URL scheme {scheme!r} is not allowed; use http or https.")

    def load(self) -> list[Document]:
        self._validate()
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
