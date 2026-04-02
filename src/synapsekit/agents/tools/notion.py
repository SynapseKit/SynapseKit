from __future__ import annotations

import asyncio
import os
import random
from typing import Any

import httpx

from ..base import BaseTool, ToolResult


class NotionTool(BaseTool):
    """Interact with Notion pages and databases."""

    name = "notion"
    description = (
        "Interact with Notion pages and databases. "
        "Operations: search, get_page, create_page, append_block. "
        "Requires NOTION_API_KEY environment variable."
    )
    parameters = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "description": "Operation to perform: search, get_page, create_page, append_block",
                "enum": ["search", "get_page", "create_page", "append_block"],
            },
            "query": {
                "type": "string",
                "description": "Search query (for search operation)",
                "default": "",
            },
            "page_id": {
                "type": "string",
                "description": "Page ID (for get_page, append_block operations)",
                "default": "",
            },
            "parent_id": {
                "type": "string",
                "description": "Parent database or page ID (for create_page operation)",
                "default": "",
            },
            "title": {
                "type": "string",
                "description": "Page title (for create_page operation)",
                "default": "",
            },
            "content": {
                "type": "string",
                "description": "Page content (for create_page, append_block operations)",
                "default": "",
            },
        },
        "required": ["operation"],
    }

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries

    def _get_api_key(self) -> str:
        api_key = self._api_key or os.environ.get("NOTION_API_KEY")
        if not api_key:
            raise ValueError(
                "NOTION_API_KEY not provided. Set NOTION_API_KEY env var or pass api_key."
            )
        return api_key

    def _get_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._get_api_key()}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json",
        }

    async def _request_with_retry(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        json_data: dict | None = None,
    ) -> httpx.Response:
        """Make HTTP request with retry logic for transient failures."""
        retryable_statuses = {409, 429, 500, 502, 503, 504}
        base_delay = 1.0
        max_wait = 60.0

        for attempt in range(self._max_retries + 1):
            try:
                if method == "GET":
                    resp = await client.get(url, headers=self._get_headers())
                elif method == "POST":
                    resp = await client.post(url, headers=self._get_headers(), json=json_data or {})
                elif method == "PATCH":
                    resp = await client.patch(
                        url, headers=self._get_headers(), json=json_data or {}
                    )
                else:
                    raise ValueError(f"Unsupported method: {method}")

                if resp.status_code < 400:
                    return resp

                if resp.status_code not in retryable_statuses:
                    resp.raise_for_status()

                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        try:
                            wait_time = float(retry_after)
                        except ValueError:
                            wait_time = min(base_delay * (2**attempt), max_wait)
                    else:
                        wait_time = min(base_delay * (2**attempt), max_wait)
                else:
                    wait_time = min(base_delay * (2**attempt), max_wait)

                wait_time += random.uniform(0, 0.5)

                if attempt < self._max_retries:
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    resp.raise_for_status()

            except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError):
                if attempt < self._max_retries:
                    wait_time = min(base_delay * (2**attempt), max_wait)
                    wait_time += random.uniform(0, 0.5)
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise

        raise RuntimeError("Request failed after all retries")

    async def run(
        self,
        operation: str = "",
        query: str = "",
        page_id: str = "",
        parent_id: str = "",
        title: str = "",
        content: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        operation = operation or kwargs.get("op", "")
        if not operation:
            return ToolResult(
                output="",
                error="No operation provided. Use: search, get_page, create_page, append_block",
            )

        try:
            if operation == "search":
                return await self.search(query=query)
            elif operation == "get_page":
                return await self.get_page(page_id=page_id)
            elif operation == "create_page":
                return await self.create_page(parent_id=parent_id, title=title, content=content)
            elif operation == "append_block":
                return await self.append_block(page_id=page_id, content=content)
            else:
                return ToolResult(output="", error=f"Unknown operation: {operation}")
        except ValueError as e:
            return ToolResult(output="", error=str(e))
        except (httpx.HTTPError, RuntimeError) as e:
            return ToolResult(output="", error=f"Notion API error: {e}")

    async def search(self, query: str = "") -> ToolResult:
        """Search pages and databases by query."""
        url = "https://api.notion.com/v1/search"
        payload: dict[str, Any] = {"filter": {"value": "page", "property": "object"}}
        if query:
            payload["query"] = query

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await self._request_with_retry(client, "POST", url, json_data=payload)
                data = resp.json()

            results = data.get("results", [])
            if not results:
                return ToolResult(output="No pages found.")

            lines = ["Found pages:"]
            for i, item in enumerate(results, 1):
                item_id = item.get("id", "")
                title_prop = item.get("title", {}) or item.get("properties", {})
                title_text = ""
                if isinstance(title_prop, dict):
                    if "name" in title_prop:
                        title_text = title_prop["name"]
                    else:
                        for val in title_prop.values():
                            if isinstance(val, dict) and "title" in val:
                                title_val = val["title"]
                                if isinstance(title_val, list) and len(title_val) > 0:
                                    title_text = title_val[0].get("plain_text", "")
                                    break
                                elif isinstance(title_val, str):
                                    title_text = title_val
                                    break
                if not title_text:
                    title_text = item.get("parent", {}).get("type", "page")
                lines.append(f"{i}. {title_text} — {item_id}")

            return ToolResult(output="\n".join(lines))
        except httpx.HTTPStatusError as e:
            return ToolResult(
                output="", error=f"Notion API error: {e.response.status_code} - {e.response.text}"
            )
        except httpx.RequestError as e:
            return ToolResult(output="", error=f"Request failed: {e}")

    async def get_page(self, page_id: str) -> ToolResult:
        """Retrieve page content by page ID."""
        if not page_id:
            return ToolResult(output="", error="No page_id provided.")

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                page_resp = await self._request_with_retry(
                    client, "GET", f"https://api.notion.com/v1/pages/{page_id}"
                )
                page_data = page_resp.json()

                title = ""
                properties = page_data.get("properties", {})
                for prop in properties.values():
                    if prop.get("type") == "title" and prop.get("title"):
                        title_list = prop.get("title", [])
                        if title_list:
                            title = title_list[0].get("plain_text", "")
                        break

                url = page_data.get("url", "")

                blocks_resp = await self._request_with_retry(
                    client, "GET", f"https://api.notion.com/v1/blocks/{page_id}/children"
                )
                blocks_data = blocks_resp.json()

            content_lines = []
            for block in blocks_data.get("results", []):
                block_type = block.get("type", "")
                block_content = block.get(block_type, {})
                if isinstance(block_content, dict):
                    rich_text = block_content.get("rich_text", [])
                    for rt in rich_text:
                        text = rt.get("plain_text", "")
                        if text:
                            content_lines.append(text)

            if not content_lines:
                content_text = "(No content)"
            else:
                content_text = "\n".join(content_lines)

            output = f"Title: {title}\n\nContent:\n{content_text}\n\nURL: {url}"
            return ToolResult(output=output)
        except httpx.HTTPStatusError as e:
            return ToolResult(
                output="", error=f"Notion API error: {e.response.status_code} - {e.response.text}"
            )
        except httpx.RequestError as e:
            return ToolResult(output="", error=f"Request failed: {e}")

    async def create_page(
        self, parent_id: str = "", title: str = "", content: str = ""
    ) -> ToolResult:
        """Create a new page in a database or as a child page."""
        if not parent_id:
            return ToolResult(output="", error="No parent_id provided.")

        url = "https://api.notion.com/v1/pages"

        parent_id_stripped = parent_id.replace("-", "")
        if len(parent_id_stripped) == 32:
            parent_payload: dict[str, Any] = {"page_id": parent_id}
        else:
            parent_payload = {"database_id": parent_id}

        payload: dict[str, Any] = {
            "parent": parent_payload,
            "properties": {},
        }

        if title:
            payload["properties"]["title"] = [{"type": "text", "text": {"content": title}}]

        blocks = []
        if content:
            lines = content.split("\n")
            for line in lines[:100]:
                blocks.append(
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {"rich_text": [{"type": "text", "text": {"content": line}}]},
                    }
                )

        if blocks:
            payload["children"] = blocks

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await self._request_with_retry(client, "POST", url, json_data=payload)
                data = resp.json()

            page_url = data.get("url", "")
            return ToolResult(output=f"Page created successfully:\n{page_url}")
        except httpx.HTTPStatusError as e:
            return ToolResult(
                output="", error=f"Notion API error: {e.response.status_code} - {e.response.text}"
            )
        except httpx.RequestError as e:
            return ToolResult(output="", error=f"Request failed: {e}")

    async def append_block(self, page_id: str = "", content: str = "") -> ToolResult:
        """Append content to an existing page."""
        if not page_id:
            return ToolResult(output="", error="No page_id provided.")
        if not content:
            return ToolResult(output="", error="No content provided.")

        url = f"https://api.notion.com/v1/blocks/{page_id}/children"

        lines = content.split("\n")
        blocks = []
        for line in lines[:100]:
            blocks.append(
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"type": "text", "text": {"content": line}}]},
                }
            )

        payload = {"children": blocks}

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                await self._request_with_retry(client, "PATCH", url, json_data=payload)

            return ToolResult(output="Content appended successfully.")
        except httpx.HTTPStatusError as e:
            return ToolResult(
                output="", error=f"Notion API error: {e.response.status_code} - {e.response.text}"
            )
        except httpx.RequestError as e:
            return ToolResult(output="", error=f"Request failed: {e}")
