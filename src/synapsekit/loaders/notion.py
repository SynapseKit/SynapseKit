"""NotionLoader — load pages or databases from the Notion API."""

from __future__ import annotations

import asyncio
import logging
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx

from .base import Document

logger = logging.getLogger(__name__)


class NotionLoader:
    """Load pages or databases from Notion into Documents.

    This loader uses the Notion API to fetch pages or database entries.
    It supports both async and synchronous loading.

    Prerequisites:
        - A Notion integration token with appropriate permissions
        - Access to the page or database you want to load

    Example::

        # Load a single page
        loader = NotionLoader(api_key="your-api-key", page_id="page-id-here")
        docs = await loader.load()

        # Load all pages from a database
        loader = NotionLoader(api_key="your-api-key", database_id="database-id-here")
        docs = await loader.load()
    """

    def __init__(
        self,
        api_key: str,
        page_id: str | None = None,
        database_id: str | None = None,
        max_retries: int = 3,
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        if not page_id and not database_id:
            raise ValueError("Either page_id or database_id must be provided")
        if page_id and database_id:
            raise ValueError("Only one of page_id or database_id can be provided")

        self.api_key = api_key
        self.page_id = page_id
        self.database_id = database_id
        self.max_retries = max_retries
        self.timeout = timeout

        self._base_url = "https://api.notion.com/v1"
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json",
        }

    async def load(self) -> list[Document]:
        """Asynchronously fetch pages and return them as Documents."""
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required: pip install synapsekit[notion]") from None

        timeout_config = httpx.Timeout(self.timeout, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            if self.page_id:
                return [await self._load_page(client, self.page_id)]
            else:
                # database_id is guaranteed to be str here due to __init__ validation
                assert self.database_id is not None
                return await self._load_database(client, self.database_id)

    async def _load_page(self, client: httpx.AsyncClient, page_id: str) -> Document:
        """Load a single page by ID."""
        # Get page metadata
        page_response = await self._request_with_retry(
            client,
            "GET",
            f"{self._base_url}/pages/{page_id}",
        )
        page_data = page_response.json()

        # Get page blocks (content)
        blocks = await self._get_block_children(client, page_id)

        # Extract text and headings
        text, headings = self._extract_text_from_blocks(blocks)

        # Build metadata
        title = self._extract_title(page_data)
        metadata = {
            "source": "notion",
            "page_id": page_id,
            "title": title,
            "url": page_data.get("url", ""),
            "headings": headings,
        }

        return Document(text=text, metadata=metadata)

    async def _load_database(self, client: httpx.AsyncClient, database_id: str) -> list[Document]:
        """Load all pages from a database."""
        # Query database for all pages
        query_response = await self._request_with_retry(
            client,
            "POST",
            f"{self._base_url}/databases/{database_id}/query",
            json_data={},
        )
        query_data = query_response.json()

        # Load each page
        documents = []
        for result in query_data.get("results", []):
            page_id = result["id"]
            doc = await self._load_page(client, page_id)
            documents.append(doc)

        return documents

    async def _get_block_children(self, client: httpx.AsyncClient, block_id: str) -> list[dict]:
        """Recursively get all block children."""
        all_blocks = []
        has_more = True
        start_cursor = None

        while has_more:
            params: dict[str, str] = {}
            if start_cursor:
                params["start_cursor"] = start_cursor

            response = await self._request_with_retry(
                client,
                "GET",
                f"{self._base_url}/blocks/{block_id}/children",
                params=params,
            )
            data = response.json()

            blocks = data.get("results", [])
            all_blocks.extend(blocks)

            has_more = data.get("has_more", False)
            start_cursor = data.get("next_cursor")
            if has_more and not start_cursor:
                break

        # Recursively get children for blocks that have them
        for block in all_blocks[:]:  # Copy list to avoid modification during iteration
            if block.get("has_children", False):
                children = await self._get_block_children(client, block["id"])
                all_blocks.extend(children)

        return all_blocks

    def _extract_text_from_blocks(self, blocks: list[dict]) -> tuple[str, list[str]]:
        """Extract plain text and headings from blocks."""
        text_parts = []
        headings = []

        for block in blocks:
            block_type = block.get("type")
            if not block_type:
                continue

            block_content = block.get(block_type, {})
            rich_text = block_content.get("rich_text", [])

            if not rich_text:
                continue

            # Extract plain text from rich_text array
            block_text = "".join(rt.get("plain_text", "") for rt in rich_text)

            if not block_text.strip():
                continue

            # Track headings
            if block_type in ("heading_1", "heading_2", "heading_3"):
                headings.append(block_text)

            text_parts.append(block_text)

        return "\n".join(text_parts), headings

    def _extract_title(self, page_data: dict) -> str:
        """Extract title from page properties."""
        properties = page_data.get("properties", {})

        # Try to find a title property
        for _prop_name, prop_value in properties.items():
            if prop_value.get("type") == "title":
                title_array = prop_value.get("title", [])
                if title_array:
                    return "".join(t.get("plain_text", "") for t in title_array)

        return "Untitled"

    async def _request_with_retry(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        params: dict | None = None,
        json_data: dict | None = None,
    ) -> httpx.Response:
        """Make HTTP request with retry logic for transient failures.

        Based on Notion API official documentation:
        https://developers.notion.com/reference/request-limits
        https://developers.notion.com/reference/status-codes

        Rate Limit: Average 3 requests/second (with bursts allowed)

        Retry decision metrics:
        - 429 (rate_limited): ALWAYS retry, MUST respect Retry-After header
        - 409 (conflict_error): Retry (data collision, try again)
        - 500 (internal_server_error): Retry
        - 502 (bad_gateway): Retry (upstream connection failed)
        - 503 (service_unavailable, database_connection_unavailable): Retry
        - 504 (gateway_timeout): Retry (Notion timeout >60s)
        - Network errors (timeouts, connection): Retry

        NO RETRY (permanent client errors):
        - 400 (invalid_json, invalid_request_url, validation_error, etc.)
        - 401 (unauthorized): Invalid API token
        - 403 (restricted_resource): No permission
        - 404 (object_not_found): Resource doesn't exist or not shared

        Retry strategy:
        - Exponential backoff: wait = min(base * (2 ** attempt) + jitter, 60s)
        - Base delay: 1 second
        - Jitter: random 0-0.5s to prevent thundering herd
        - Max retries: configurable (default 3)
        - Retry-After header: Takes precedence for 429 responses
        """
        import httpx

        last_exception = None
        base_delay = 1.0
        max_wait = 60.0

        # Status codes that should be retried
        retryable_statuses = {409, 429, 500, 502, 503, 504}

        for attempt in range(self.max_retries + 1):
            try:
                # Make the request
                if method == "GET":
                    response = await client.get(url, headers=self._headers, params=params or {})
                elif method == "POST":
                    response = await client.post(
                        url,
                        headers=self._headers,
                        params=params or {},
                        json=json_data or {},
                    )
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                # Success - return immediately
                if response.status_code < 400:
                    return response

                # Check if this status code is retryable
                if response.status_code not in retryable_statuses:
                    # Permanent error (4xx except 409/429), don't retry
                    response.raise_for_status()

                # Handle specific retryable errors
                if response.status_code == 429:
                    # Rate limited - MUST respect Retry-After header
                    if attempt < self.max_retries:
                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            try:
                                wait_time = float(retry_after)
                            except ValueError:
                                wait_time = min(base_delay * (2**attempt), max_wait)
                        else:
                            wait_time = min(base_delay * (2**attempt), max_wait)

                        jitter = random.uniform(0, 0.5)
                        wait_time += jitter

                        logger.warning(
                            f"Notion API rate limit (429). "
                            f"Retrying in {wait_time:.2f}s "
                            f"(attempt {attempt + 1}/{self.max_retries})"
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        response.raise_for_status()

                elif response.status_code == 409:
                    # Conflict error - data collision, retry may succeed
                    if attempt < self.max_retries:
                        wait_time = min(base_delay * (2**attempt), max_wait)
                        jitter = random.uniform(0, 0.5)
                        wait_time += jitter

                        logger.warning(
                            f"Notion API conflict (409). "
                            f"Retrying in {wait_time:.2f}s "
                            f"(attempt {attempt + 1}/{self.max_retries})"
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        response.raise_for_status()

                elif response.status_code in {500, 502, 503, 504}:
                    # Server errors - retry with exponential backoff
                    if attempt < self.max_retries:
                        wait_time = min(base_delay * (2**attempt), max_wait)
                        jitter = random.uniform(0, 0.5)
                        wait_time += jitter

                        logger.warning(
                            f"Notion API server error "
                            f"({response.status_code}). "
                            f"Retrying in {wait_time:.2f}s "
                            f"(attempt {attempt + 1}/{self.max_retries})"
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        response.raise_for_status()

                # Shouldn't reach here, but raise if we do
                response.raise_for_status()

            except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as e:
                # Network errors - retry with exponential backoff
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = min(base_delay * (2**attempt), max_wait)
                    jitter = random.uniform(0, 0.5)
                    wait_time += jitter

                    logger.warning(
                        f"Network error: {type(e).__name__}. "
                        f"Retrying in {wait_time:.2f}s "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise

            except httpx.HTTPStatusError:
                # HTTP errors not caught above - should only be non-retryable ones
                raise

        # If we exhausted all retries
        if last_exception:
            raise last_exception
        raise RuntimeError("Request failed after all retries")
