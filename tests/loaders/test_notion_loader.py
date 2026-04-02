"""Tests for NotionLoader."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synapsekit.loaders.notion import NotionLoader


class TestNotionLoaderValidation:
    def test_api_key_required(self):
        with pytest.raises(ValueError, match="api_key is required"):
            NotionLoader(api_key="")

    def test_page_or_database_id_required(self):
        with pytest.raises(ValueError, match="Either page_id or database_id must be provided"):
            NotionLoader(api_key="test-key")

    def test_cannot_provide_both_page_and_database_id(self):
        with pytest.raises(ValueError, match="Only one of page_id or database_id can be provided"):
            NotionLoader(api_key="test-key", page_id="page-123", database_id="db-456")

    def test_valid_construction_with_page_id(self):
        loader = NotionLoader(api_key="test-key", page_id="page-123")
        assert loader.api_key == "test-key"
        assert loader.page_id == "page-123"
        assert loader.database_id is None

    def test_valid_construction_with_database_id(self):
        loader = NotionLoader(api_key="test-key", database_id="db-123")
        assert loader.api_key == "test-key"
        assert loader.database_id == "db-123"
        assert loader.page_id is None


class TestNotionLoaderImport:
    async def test_missing_httpx_raises_import_error(self):
        import sys

        loader = NotionLoader(api_key="test-key", page_id="page-123")
        with patch.dict(sys.modules, {"httpx": None}):
            with pytest.raises(ImportError, match="httpx required"):
                await loader.load()


class TestNotionLoaderPage:
    def _mock_page_response(self, page_id: str = "page-123", title: str = "Test Page"):
        """Mock a Notion page metadata response."""
        return {
            "id": page_id,
            "url": f"https://notion.so/{page_id}",
            "properties": {
                "Title": {
                    "type": "title",
                    "title": [{"plain_text": title}],
                }
            },
        }

    def _mock_blocks_response(self, blocks: list[dict] | None = None):
        """Mock a Notion blocks response."""
        if blocks is None:
            blocks = []
        return {
            "results": blocks,
            "has_more": False,
            "next_cursor": None,
        }

    def _make_block(self, block_type: str, text: str, has_children: bool = False) -> dict:
        """Create a mock Notion block."""
        return {
            "id": f"block-{block_type}-id",
            "type": block_type,
            "has_children": has_children,
            block_type: {
                "rich_text": [{"plain_text": text}],
            },
        }

    def _mock_retry_method(self, responses: list[dict]):
        """Create a mock for _request_with_retry that returns responses in sequence."""
        response_iter = iter(responses)

        async def mock_request(*args, **kwargs):
            data = next(response_iter)
            mock_resp = MagicMock()
            mock_resp.json = lambda: data
            return mock_resp

        return mock_request

    async def test_load_single_page(self):
        """Test loading a single page."""
        loader = NotionLoader(api_key="test-key", page_id="page-123")

        page_data = self._mock_page_response()
        blocks_data = self._mock_blocks_response(
            [
                self._make_block("heading_1", "My Heading"),
                self._make_block("paragraph", "Some text content"),
            ]
        )

        # Mock responses in order: page metadata, blocks
        mock_retry = self._mock_retry_method([page_data, blocks_data])

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            with patch("httpx.Timeout"):
                with patch.object(loader, "_request_with_retry", side_effect=mock_retry):
                    docs = await loader.load()

        assert len(docs) == 1
        assert docs[0].text == "My Heading\nSome text content"
        assert docs[0].metadata["source"] == "notion"
        assert docs[0].metadata["page_id"] == "page-123"
        assert docs[0].metadata["title"] == "Test Page"
        assert docs[0].metadata["headings"] == ["My Heading"]

    async def test_load_page_with_multiple_headings(self):
        """Test that headings are properly extracted."""
        loader = NotionLoader(api_key="test-key", page_id="page-123")

        page_data = self._mock_page_response()
        blocks_data = self._mock_blocks_response(
            [
                self._make_block("heading_1", "Main Title"),
                self._make_block("paragraph", "Intro text"),
                self._make_block("heading_2", "Subtitle"),
                self._make_block("paragraph", "More content"),
                self._make_block("heading_3", "Sub-subtitle"),
            ]
        )

        mock_retry = self._mock_retry_method([page_data, blocks_data])

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            with patch("httpx.Timeout"):
                with patch.object(loader, "_request_with_retry", side_effect=mock_retry):
                    docs = await loader.load()

        assert len(docs[0].metadata["headings"]) == 3
        assert docs[0].metadata["headings"] == ["Main Title", "Subtitle", "Sub-subtitle"]

    async def test_load_page_with_various_block_types(self):
        """Test extraction of different block types."""
        loader = NotionLoader(api_key="test-key", page_id="page-123")

        page_data = self._mock_page_response()
        blocks_data = self._mock_blocks_response(
            [
                self._make_block("paragraph", "Regular paragraph"),
                self._make_block("bulleted_list_item", "Bullet point"),
                self._make_block("numbered_list_item", "Numbered item"),
                self._make_block("to_do", "Todo item"),
                self._make_block("code", "print('hello')"),
            ]
        )

        mock_retry = self._mock_retry_method([page_data, blocks_data])

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            with patch("httpx.Timeout"):
                with patch.object(loader, "_request_with_retry", side_effect=mock_retry):
                    docs = await loader.load()

        expected_text = "Regular paragraph\nBullet point\nNumbered item\nTodo item\nprint('hello')"
        assert docs[0].text == expected_text

    async def test_load_page_with_pagination(self):
        """Test loading blocks with pagination."""
        loader = NotionLoader(api_key="test-key", page_id="page-123")

        page_data = self._mock_page_response()

        # First page of blocks
        blocks_page1 = {
            "results": [self._make_block("paragraph", "First block")],
            "has_more": True,
            "next_cursor": "cursor-123",
        }

        # Second page of blocks
        blocks_page2 = {
            "results": [self._make_block("paragraph", "Second block")],
            "has_more": False,
            "next_cursor": None,
        }

        mock_retry = self._mock_retry_method([page_data, blocks_page1, blocks_page2])

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            with patch("httpx.Timeout"):
                with patch.object(loader, "_request_with_retry", side_effect=mock_retry):
                    docs = await loader.load()

        assert docs[0].text == "First block\nSecond block"

    async def test_load_page_with_nested_blocks(self):
        """Test loading blocks with children."""
        loader = NotionLoader(api_key="test-key", page_id="page-123")

        page_data = self._mock_page_response()

        # Parent block has children
        parent_block = self._make_block("paragraph", "Parent", has_children=True)
        child_block = self._make_block("paragraph", "Child")

        # First call returns parent with has_children=True
        blocks_response = {
            "results": [parent_block],
            "has_more": False,
            "next_cursor": None,
        }

        # Second call returns children
        children_response = {
            "results": [child_block],
            "has_more": False,
            "next_cursor": None,
        }

        mock_retry = self._mock_retry_method([page_data, blocks_response, children_response])

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            with patch("httpx.Timeout"):
                with patch.object(loader, "_request_with_retry", side_effect=mock_retry):
                    docs = await loader.load()

        assert docs[0].text == "Parent\nChild"

    async def test_load_page_without_title(self):
        """Test page without a title property."""
        loader = NotionLoader(api_key="test-key", page_id="page-123")

        page_data = {"id": "page-123", "url": "https://notion.so/page-123", "properties": {}}
        blocks_data = self._mock_blocks_response([self._make_block("paragraph", "Content")])

        mock_retry = self._mock_retry_method([page_data, blocks_data])

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            with patch("httpx.Timeout"):
                with patch.object(loader, "_request_with_retry", side_effect=mock_retry):
                    docs = await loader.load()

        assert docs[0].metadata["title"] == "Untitled"


class TestNotionLoaderDatabase:
    async def test_load_database(self):
        """Test loading pages from a database."""
        loader = NotionLoader(api_key="test-key", database_id="db-123")

        # Database query response
        db_query_response = {
            "results": [
                {"id": "page-1"},
                {"id": "page-2"},
            ]
        }

        # Mock page responses
        page1_data = {
            "id": "page-1",
            "url": "https://notion.so/page-1",
            "properties": {"Title": {"type": "title", "title": [{"plain_text": "Page 1"}]}},
        }
        page2_data = {
            "id": "page-2",
            "url": "https://notion.so/page-2",
            "properties": {"Title": {"type": "title", "title": [{"plain_text": "Page 2"}]}},
        }

        blocks1 = {
            "results": [
                {
                    "id": "b1",
                    "type": "paragraph",
                    "has_children": False,
                    "paragraph": {"rich_text": [{"plain_text": "Content 1"}]},
                }
            ],
            "has_more": False,
        }
        blocks2 = {
            "results": [
                {
                    "id": "b2",
                    "type": "paragraph",
                    "has_children": False,
                    "paragraph": {"rich_text": [{"plain_text": "Content 2"}]},
                }
            ],
            "has_more": False,
        }

        # Responses in order: db query, page1 metadata, page1 blocks, page2 metadata, page2 blocks
        responses = [db_query_response, page1_data, blocks1, page2_data, blocks2]
        response_iter = iter(responses)

        async def mock_request(*args, **kwargs):
            data = next(response_iter)
            mock_resp = MagicMock()
            mock_resp.json = lambda: data
            return mock_resp

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            with patch("httpx.Timeout"):
                with patch.object(loader, "_request_with_retry", side_effect=mock_request):
                    docs = await loader.load()

        assert len(docs) == 2
        assert docs[0].text == "Content 1"
        assert docs[0].metadata["title"] == "Page 1"
        assert docs[0].metadata["page_id"] == "page-1"
        assert docs[1].text == "Content 2"
        assert docs[1].metadata["title"] == "Page 2"
        assert docs[1].metadata["page_id"] == "page-2"

    async def test_load_empty_database(self):
        """Test loading a database with no pages."""
        loader = NotionLoader(api_key="test-key", database_id="db-123")

        db_query_response = {"results": []}

        async def mock_request(*args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.json = lambda: db_query_response
            return mock_resp

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            with patch("httpx.Timeout"):
                with patch.object(loader, "_request_with_retry", side_effect=mock_request):
                    docs = await loader.load()

        assert docs == []
