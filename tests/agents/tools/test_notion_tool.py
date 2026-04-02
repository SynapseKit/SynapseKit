from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest

from synapsekit.agents.tools.notion import NotionTool


class MockResponse:
    def __init__(
        self,
        status_code: int,
        json_data: dict,
        text: str = "",
        headers: dict | None = None,
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data
        self.text = text or str(json_data)
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if 400 <= self.status_code < 600:
            import httpx

            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=AsyncMock(),
                response=self,
            )

    def json(self) -> dict:
        return self._json_data


@pytest.fixture
def notion_api_key() -> str:
    return "test_notion_api_key_12345"


@pytest.fixture
def mock_httpx_client(notion_api_key: str):
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    return mock_client


async def test_search_success(notion_api_key: str, mock_httpx_client: AsyncMock) -> None:
    mock_response = MockResponse(
        200,
        {
            "results": [
                {
                    "id": "page-123",
                    "properties": {"Name": {"title": [{"plain_text": "Project Notes"}]}},
                },
                {
                    "id": "page-456",
                    "properties": {"Name": {"title": [{"plain_text": "Meeting Notes"}]}},
                },
            ]
        },
    )
    mock_httpx_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        with patch.dict(os.environ, {"NOTION_API_KEY": notion_api_key}):
            tool = NotionTool()
            result = await tool.run(operation="search", query="notes")

    assert result.error is None
    assert "Found pages:" in result.output
    assert "Project Notes" in result.output
    assert "Meeting Notes" in result.output
    assert "page-123" in result.output


async def test_search_no_results(notion_api_key: str, mock_httpx_client: AsyncMock) -> None:
    mock_response = MockResponse(200, {"results": []})
    mock_httpx_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        with patch.dict(os.environ, {"NOTION_API_KEY": notion_api_key}):
            tool = NotionTool()
            result = await tool.run(operation="search", query="nonexistent")

    assert result.error is None
    assert "No pages found" in result.output


async def test_get_page_success(notion_api_key: str, mock_httpx_client: AsyncMock) -> None:
    page_response = MockResponse(
        200,
        {
            "id": "page-123",
            "url": "https://notion.so/page-123",
            "properties": {"Name": {"type": "title", "title": [{"plain_text": "Project Notes"}]}},
        },
    )
    blocks_response = MockResponse(
        200,
        {
            "results": [
                {
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"plain_text": "This is the first paragraph."}]},
                },
                {
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"plain_text": "This is the second paragraph."}]},
                },
            ]
        },
    )
    mock_httpx_client.get = AsyncMock(side_effect=[page_response, blocks_response])

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        with patch.dict(os.environ, {"NOTION_API_KEY": notion_api_key}):
            tool = NotionTool()
            result = await tool.run(operation="get_page", page_id="page-123")

    assert result.error is None
    assert "Title: Project Notes" in result.output
    assert "This is the first paragraph." in result.output
    assert "https://notion.so/page-123" in result.output


async def test_get_page_no_page_id(notion_api_key: str) -> None:
    with patch.dict(os.environ, {"NOTION_API_KEY": notion_api_key}):
        tool = NotionTool()
        result = await tool.run(operation="get_page")

    assert result.error is not None
    assert "No page_id provided" in result.error


async def test_create_page_success(notion_api_key: str, mock_httpx_client: AsyncMock) -> None:
    mock_response = MockResponse(
        200,
        {"id": "new-page-789", "url": "https://notion.so/new-page-789"},
    )
    mock_httpx_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        with patch.dict(os.environ, {"NOTION_API_KEY": notion_api_key}):
            tool = NotionTool()
            result = await tool.run(
                operation="create_page",
                parent_id="db-123",
                title="New Page",
                content="Page content here",
            )

    assert result.error is None
    assert "Page created successfully" in result.output
    assert "https://notion.so/new-page-789" in result.output


async def test_create_page_no_parent_id(notion_api_key: str) -> None:
    with patch.dict(os.environ, {"NOTION_API_KEY": notion_api_key}):
        tool = NotionTool()
        result = await tool.run(operation="create_page", title="Test")

    assert result.error is not None
    assert "No parent_id provided" in result.error


async def test_append_block_success(notion_api_key: str, mock_httpx_client: AsyncMock) -> None:
    mock_response = MockResponse(200, {"results": []})
    mock_httpx_client.patch = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        with patch.dict(os.environ, {"NOTION_API_KEY": notion_api_key}):
            tool = NotionTool()
            result = await tool.run(
                operation="append_block", page_id="page-123", content="New content to append"
            )

    assert result.error is None
    assert "Content appended successfully" in result.output


async def test_append_block_no_page_id(notion_api_key: str) -> None:
    with patch.dict(os.environ, {"NOTION_API_KEY": notion_api_key}):
        tool = NotionTool()
        result = await tool.run(operation="append_block", content="test")

    assert result.error is not None
    assert "No page_id provided" in result.error


async def test_append_block_no_content(notion_api_key: str) -> None:
    with patch.dict(os.environ, {"NOTION_API_KEY": notion_api_key}):
        tool = NotionTool()
        result = await tool.run(operation="append_block", page_id="page-123")

    assert result.error is not None
    assert "No content provided" in result.error


async def test_no_operation_provided(notion_api_key: str) -> None:
    with patch.dict(os.environ, {"NOTION_API_KEY": notion_api_key}):
        tool = NotionTool()
        result = await tool.run()

    assert result.error is not None
    assert "No operation provided" in result.error


async def test_unknown_operation(notion_api_key: str) -> None:
    with patch.dict(os.environ, {"NOTION_API_KEY": notion_api_key}):
        tool = NotionTool()
        result = await tool.run(operation="invalid_op")

    assert result.error is not None
    assert "Unknown operation" in result.error


async def test_missing_api_key() -> None:
    with patch.dict(os.environ, {}, clear=True):
        tool = NotionTool()
        result = await tool.run(operation="search", query="test")

    assert result.error is not None
    assert "NOTION_API_KEY not provided" in result.error


async def test_api_key_from_init() -> None:
    tool = NotionTool(api_key="init_key_123")
    assert tool._get_api_key() == "init_key_123"


async def test_http_status_error(notion_api_key: str, mock_httpx_client: AsyncMock) -> None:
    mock_response = MockResponse(401, {"code": "unauthorized", "message": "Invalid API key"})
    mock_httpx_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        with patch.dict(os.environ, {"NOTION_API_KEY": notion_api_key}):
            tool = NotionTool()
            result = await tool.run(operation="search", query="test")

    assert result.error is not None
    assert "Notion API error" in result.error


async def test_search_with_empty_query(notion_api_key: str, mock_httpx_client: AsyncMock) -> None:
    mock_response = MockResponse(
        200,
        {
            "results": [
                {"id": "page-123", "properties": {"Name": {"title": [{"plain_text": "Untitled"}]}}}
            ]
        },
    )
    mock_httpx_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        with patch.dict(os.environ, {"NOTION_API_KEY": notion_api_key}):
            tool = NotionTool()
            result = await tool.run(operation="search")

    assert result.error is None
    assert "Found pages:" in result.output


async def test_retry_on_rate_limit(notion_api_key: str, mock_httpx_client: AsyncMock) -> None:
    """Test that the tool retries on 429 rate limit error."""
    rate_limit_response = MockResponse(
        429, {"code": "rate_limited"}, headers={"Retry-After": "0.01"}
    )
    success_response = MockResponse(200, {"results": []})

    mock_httpx_client.post = AsyncMock(side_effect=[rate_limit_response, success_response])

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        with patch.dict(os.environ, {"NOTION_API_KEY": notion_api_key}):
            tool = NotionTool(max_retries=3)
            result = await tool.run(operation="search", query="test")

    assert result.error is None


async def test_retry_on_server_error(notion_api_key: str, mock_httpx_client: AsyncMock) -> None:
    """Test that the tool retries on 500 server error."""
    error_response = MockResponse(500, {"code": "internal_server_error"})
    success_response = MockResponse(200, {"results": []})

    mock_httpx_client.post = AsyncMock(side_effect=[error_response, success_response])

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        with patch.dict(os.environ, {"NOTION_API_KEY": notion_api_key}):
            tool = NotionTool(max_retries=3)
            result = await tool.run(operation="search", query="test")

    assert result.error is None


async def test_no_retry_on_auth_error(notion_api_key: str, mock_httpx_client: AsyncMock) -> None:
    """Test that the tool does not retry on 401 auth error."""
    error_response = MockResponse(401, {"code": "unauthorized"})
    mock_httpx_client.post = AsyncMock(return_value=error_response)

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        with patch.dict(os.environ, {"NOTION_API_KEY": notion_api_key}):
            tool = NotionTool(max_retries=3)
            result = await tool.run(operation="search", query="test")

    assert result.error is not None
    assert "401" in result.error
    assert mock_httpx_client.post.call_count == 1
