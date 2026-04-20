from __future__ import annotations

import asyncio
from collections import deque
from typing import Any
from urllib.parse import urljoin, urlparse

from .base import Document

_MAX_TEXT_LENGTH = 100_000
_MAX_CONCURRENCY = 10
_DEFAULT_TIMEOUT = 30


def _extract_text(html: str) -> str:
    # Best-effort HTML to text extraction
    from bs4 import BeautifulSoup

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)[:_MAX_TEXT_LENGTH]


def _parse_sitemap(xml: str, base_url: str) -> tuple[list[dict[str, str]], list[str]]:
    """Return (url_entries, sitemap_index_urls).

    *url_entries* — list of ``{"loc": ..., "lastmod": ...}`` dicts from ``<url>`` tags.
    *sitemap_index_urls* — list of child sitemap URLs from ``<sitemap>`` tags.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(xml, "lxml-xml")

    # Sitemap index — contains nested <sitemap> entries
    index_urls = [
        urljoin(base_url, loc_tag.get_text(strip=True))
        for tag in soup.find_all("sitemap")
        if (loc_tag := tag.find("loc"))
    ]

    # Regular sitemap — contains <url> entries
    url_entries: list[dict[str, str]] = []
    for tag in soup.find_all("url"):
        loc_tag = tag.find("loc")
        if not loc_tag:
            continue
        lastmod_tag = tag.find("lastmod")
        url_entries.append(
            {
                "loc": loc_tag.get_text(strip=True),
                "lastmod": lastmod_tag.get_text(strip=True) if lastmod_tag else "",
            }
        )

    return url_entries, index_urls


class SitemapLoader:
    """Discover and load pages from a website's sitemap.xml.

    Supports both regular sitemaps and sitemap index files (nested sitemaps).
    Each discovered page is fetched and its text is extracted as a
    :class:`Document` with ``source``, ``url``, and ``lastmod`` metadata.

    Parameters
    ----------
    url : str
        URL of the sitemap — typically ``https://example.com/sitemap.xml``.
    limit : int | None
        Maximum number of pages to fetch (applied after URL discovery).
    filter_urls : list[str] | None
        Only load URLs that contain one of these substrings.
    """

    def __init__(
        self,
        url: str,
        limit: int | None = None,
        filter_urls: list[str] | None = None,
    ) -> None:
        if not url:
            raise ValueError("url must be provided")
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"URL scheme {parsed.scheme!r} is not allowed; use http or https.")

        self._url = url
        self._limit = limit
        self._filter_urls = filter_urls

    def load(self) -> list[Document]:
        """Synchronous entry point."""
        return asyncio.run(self.aload())

    async def aload(self) -> list[Document]:
        """Discover sitemap URLs then fetch pages concurrently in batches."""
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required: pip install synapsekit[web]") from None

        url_entries = await self._collect_urls(httpx)

        if self._filter_urls:
            url_entries = [e for e in url_entries if any(f in e["loc"] for f in self._filter_urls)]

        if self._limit is not None:
            url_entries = url_entries[: self._limit]

        if not url_entries:
            return []

        docs: list[Document] = []
        async with httpx.AsyncClient(follow_redirects=True, timeout=_DEFAULT_TIMEOUT) as client:
            for i in range(0, len(url_entries), _MAX_CONCURRENCY):
                batch = url_entries[i : i + _MAX_CONCURRENCY]
                tasks = [self._fetch_page(client, entry) for entry in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    # Skip failed pages (network errors, parsing issues)
                    # NOTE: failures are intentionally ignored for robustness
                    if isinstance(result, Exception):
                        continue
                    if isinstance(result, Document):
                        docs.append(result)

        return docs

    async def _collect_urls(self, httpx: Any) -> list[dict[str, str]]:
        """Fetch and parse sitemaps, following sitemap index files."""
        base_url = "{0.scheme}://{0.netloc}".format(urlparse(self._url))
        # deque gives O(1) popleft vs O(n) for list.pop(0)
        to_visit: deque[str] = deque([self._url])
        visited: set[str] = set()
        seen: set[str] = set()
        all_entries: list[dict[str, str]] = []

        async with httpx.AsyncClient(follow_redirects=True, timeout=_DEFAULT_TIMEOUT) as client:
            while to_visit:
                sitemap_url = to_visit.popleft()
                if sitemap_url in visited:
                    continue
                visited.add(sitemap_url)

                try:
                    resp = await client.get(sitemap_url)
                    resp.raise_for_status()
                except Exception:
                    # Skip invalid or unreachable sitemap URLs
                    continue

                url_entries, index_urls = _parse_sitemap(resp.text, base_url)
                for e in url_entries:
                    loc = e.get("loc")
                    if loc and loc not in seen:
                        seen.add(loc)
                        all_entries.append(e)
                # Queue unvisited child sitemaps
                to_visit.extend(u for u in index_urls if u not in visited)

        return all_entries

    async def _fetch_page(self, client: Any, entry: dict[str, str]) -> Document | None:
        """Fetch a single page and return a Document, or None on failure."""
        url = entry.get("loc")
        if not url:
            return None
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            text = _extract_text(resp.text)
            if not text:
                return None
            return Document(
                text=text,
                metadata={
                    "source": "sitemap",
                    "url": url,
                    "lastmod": entry.get("lastmod", ""),
                },
            )
        except Exception:
            return None
