from __future__ import annotations

import asyncio
import time
from typing import Any

from .base import Document


class ConfluenceLoader:
    """Load pages from Atlassian Confluence."""

    def __init__(
        self,
        url: str,
        username: str,
        api_token: str,
        space_key: str | None = None,
        page_id: str | None = None,
        limit: int | None = None,
    ) -> None:
        if not space_key and not page_id:
            raise ValueError("Either space_key or page_id must be provided")

        self._url = url.rstrip("/")
        self._username = username
        self._api_token = api_token
        self._space_key = space_key
        self._page_id = page_id
        self._limit = limit
        self._confluence = None

    def _init_client(self) -> Any:
        if self._confluence is None:
            try:
                from atlassian import Confluence
            except ImportError:
                raise ImportError(
                    "atlassian-python-api required: pip install synapsekit[confluence]"
                ) from None
            self._confluence = Confluence(
                url=self._url, username=self._username, password=self._api_token
            )
        return self._confluence

    def _convert_storage_to_text(self, storage_html: str) -> str:
        """Convert Confluence storage format (HTML/XML) to plain text."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "beautifulsoup4 required: pip install synapsekit[confluence]"
            ) from None

        soup = BeautifulSoup(storage_html, "html.parser")
        return str(soup.get_text(separator="\n", strip=True))

    async def _fetch_page_content_async(self, page_id: str) -> dict[str, Any]:
        """Fetch a single page with retries (async)."""
        confluence = self._init_client()
        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                loop = asyncio.get_running_loop()
                page: dict[str, Any] = await loop.run_in_executor(
                    None,
                    lambda: confluence.get_page_by_id(
                        page_id=page_id,
                        expand="body.storage,version,space,history.lastUpdated",
                    ),
                )
                return page
            except Exception as e:
                error_msg = str(e).lower()
                error_codes = ["401", "403", "404", "unauthorized", "forbidden"]
                if any(x in error_msg for x in error_codes):
                    raise RuntimeError(f"Failed to fetch page {page_id}: {e}") from e

                if attempt < max_retries - 1:
                    delay = retry_delay * (2**attempt)
                    if "429" in error_msg or "rate limit" in error_msg:
                        delay = min(delay, 60)
                    await asyncio.sleep(delay)
                else:
                    raise RuntimeError(f"Failed to fetch page {page_id} after retries: {e}") from e
        raise RuntimeError(f"Failed to fetch page {page_id}")

    def _fetch_page_content_sync(self, page_id: str) -> dict[str, Any]:
        """Fetch a single page with retries (sync)."""
        confluence = self._init_client()
        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                page: dict[str, Any] = confluence.get_page_by_id(
                    page_id=page_id,
                    expand="body.storage,version,space,history.lastUpdated",
                )
                return page
            except Exception as e:
                error_msg = str(e).lower()
                error_codes = ["401", "403", "404", "unauthorized", "forbidden"]
                if any(x in error_msg for x in error_codes):
                    raise RuntimeError(f"Failed to fetch page {page_id}: {e}") from e

                if attempt < max_retries - 1:
                    delay = retry_delay * (2**attempt)
                    if "429" in error_msg or "rate limit" in error_msg:
                        delay = min(delay, 60)
                    time.sleep(delay)
                else:
                    raise RuntimeError(f"Failed to fetch page {page_id} after retries: {e}") from e
        raise RuntimeError(f"Failed to fetch page {page_id}")

    async def _get_all_pages_in_space_async(self, space_key: str) -> list[dict[str, Any]]:
        """Get all pages in a space with automatic pagination (async)."""
        confluence = self._init_client()
        all_pages = []
        start = 0
        limit = 100

        while True:
            try:
                loop = asyncio.get_running_loop()

                def _fetch_pages(start_val: int) -> Any:
                    return confluence.get_all_pages_from_space(
                        space=space_key,
                        start=start_val,
                        limit=limit,
                        expand="body.storage,version,space,history.lastUpdated",
                    )

                result = await loop.run_in_executor(None, _fetch_pages, start)

                if not result:
                    break

                all_pages.extend(result)

                if self._limit and len(all_pages) >= self._limit:
                    all_pages = all_pages[: self._limit]
                    break

                if len(result) < limit:
                    break

                start += limit

            except Exception as e:
                error_msg = str(e).lower()
                error_codes = ["401", "403", "404", "unauthorized", "forbidden"]
                if any(x in error_msg for x in error_codes):
                    raise RuntimeError(f"Failed to fetch pages from space {space_key}: {e}") from e
                raise RuntimeError(f"Error fetching pages from space {space_key}: {e}") from e

        return all_pages

    def _get_all_pages_in_space_sync(self, space_key: str) -> list[dict[str, Any]]:
        """Get all pages in a space with automatic pagination (sync)."""
        confluence = self._init_client()
        all_pages = []
        start = 0
        limit = 100

        while True:
            try:
                result = confluence.get_all_pages_from_space(
                    space=space_key,
                    start=start,
                    limit=limit,
                    expand="body.storage,version,space,history.lastUpdated",
                )

                if not result:
                    break

                all_pages.extend(result)

                if self._limit and len(all_pages) >= self._limit:
                    all_pages = all_pages[: self._limit]
                    break

                if len(result) < limit:
                    break

                start += limit

            except Exception as e:
                error_msg = str(e).lower()
                error_codes = ["401", "403", "404", "unauthorized", "forbidden"]
                if any(x in error_msg for x in error_codes):
                    raise RuntimeError(f"Failed to fetch pages from space {space_key}: {e}") from e
                raise RuntimeError(f"Error fetching pages from space {space_key}: {e}") from e

        return all_pages

    def _page_to_document(self, page: dict[str, Any]) -> Document:
        """Convert a Confluence page to a Document."""
        # Extract page body
        body_storage = page.get("body", {}).get("storage", {}).get("value", "")
        text = self._convert_storage_to_text(body_storage)

        # Extract metadata
        metadata = {
            "source": "confluence",
            "title": page.get("title", ""),
            "page_id": page.get("id", ""),
            "space": page.get("space", {}).get("key", ""),
            "url": f"{self._url}{page.get('_links', {}).get('webui', '')}",
            "version": page.get("version", {}).get("number", ""),
        }

        # Add author if available
        if "history" in page:
            last_updated = page["history"].get("lastUpdated", {})
            if last_updated:
                metadata["author"] = last_updated.get("by", {}).get("displayName", "")
                metadata["last_modified"] = last_updated.get("when", "")

        return Document(text=text, metadata=metadata)

    async def aload(self) -> list[Document]:
        """Load pages from Confluence and return as Documents (async)."""
        if self._page_id:
            page = await self._fetch_page_content_async(self._page_id)
            return [self._page_to_document(page)]

        pages = await self._get_all_pages_in_space_async(
            self._space_key  # type: ignore[arg-type]
        )
        return [self._page_to_document(page) for page in pages]

    def load(self) -> list[Document]:
        """Load pages from Confluence and return as Documents (sync)."""
        if self._page_id:
            page = self._fetch_page_content_sync(self._page_id)
            return [self._page_to_document(page)]

        pages = self._get_all_pages_in_space_sync(
            self._space_key  # type: ignore[arg-type]
        )
        return [self._page_to_document(page) for page in pages]
