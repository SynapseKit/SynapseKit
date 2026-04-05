"""Tests for ConfluenceLoader."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from synapsekit.loaders.confluence import ConfluenceLoader


class TestConfluenceLoader:
    def test_requires_space_key_or_page_id(self):
        with pytest.raises(ValueError, match="Either space_key or page_id"):
            ConfluenceLoader(
                url="https://test.atlassian.net",
                username="user@example.com",
                api_token="token",
            )

    def test_import_error_without_atlassian_api(self):
        with patch.dict("sys.modules", {"atlassian": None}):
            loader = ConfluenceLoader(
                url="https://test.atlassian.net",
                username="user@example.com",
                api_token="token",
                page_id="123",
            )
            with pytest.raises(ImportError, match="atlassian-python-api"):
                loader.load()

    def test_load_single_page(self):
        mock_confluence_instance = MagicMock()
        mock_page = {
            "id": "123",
            "title": "Test Page",
            "body": {"storage": {"value": "<p>Test content</p>"}},
            "space": {"key": "TEST"},
            "version": {"number": 1},
            "_links": {"webui": "/spaces/TEST/pages/123"},
            "history": {
                "lastUpdated": {
                    "by": {"displayName": "John Doe"},
                    "when": "2024-01-01T00:00:00Z",
                }
            },
        }
        mock_confluence_instance.get_page_by_id.return_value = mock_page

        mock_confluence_class = MagicMock(return_value=mock_confluence_instance)
        mock_atlassian = MagicMock()
        mock_atlassian.Confluence = mock_confluence_class

        mock_soup = MagicMock()
        mock_soup.get_text.return_value = "Test content"
        mock_bs4 = MagicMock()
        mock_bs4.BeautifulSoup.return_value = mock_soup

        with patch.dict("sys.modules", {"atlassian": mock_atlassian, "bs4": mock_bs4}):
            loader = ConfluenceLoader(
                url="https://test.atlassian.net",
                username="user@example.com",
                api_token="token",
                page_id="123",
            )
            docs = loader.load()

        assert len(docs) == 1
        assert docs[0].text == "Test content"
        assert docs[0].metadata["title"] == "Test Page"
        assert docs[0].metadata["page_id"] == "123"
        assert docs[0].metadata["space"] == "TEST"
        assert docs[0].metadata["source"] == "confluence"
        assert docs[0].metadata["version"] == 1
        assert docs[0].metadata["author"] == "John Doe"
        assert "test.atlassian.net" in docs[0].metadata["url"]

    @pytest.mark.asyncio
    async def test_aload_single_page(self):
        mock_confluence_instance = MagicMock()
        mock_page = {
            "id": "123",
            "title": "Test Page Async",
            "body": {"storage": {"value": "<p>Async content</p>"}},
            "space": {"key": "TEST"},
            "version": {"number": 1},
            "_links": {"webui": "/spaces/TEST/pages/123"},
            "history": {"lastUpdated": {}},
        }
        mock_confluence_instance.get_page_by_id.return_value = mock_page

        mock_confluence_class = MagicMock(return_value=mock_confluence_instance)
        mock_atlassian = MagicMock()
        mock_atlassian.Confluence = mock_confluence_class

        mock_soup = MagicMock()
        mock_soup.get_text.return_value = "Async content"
        mock_bs4 = MagicMock()
        mock_bs4.BeautifulSoup.return_value = mock_soup

        with patch.dict("sys.modules", {"atlassian": mock_atlassian, "bs4": mock_bs4}):
            loader = ConfluenceLoader(
                url="https://test.atlassian.net",
                username="user@example.com",
                api_token="token",
                page_id="123",
            )
            docs = await loader.aload()

        assert len(docs) == 1
        assert docs[0].text == "Async content"
        assert docs[0].metadata["title"] == "Test Page Async"

    def test_load_space_pages(self):
        mock_confluence_instance = MagicMock()
        mock_pages = [
            {
                "id": "1",
                "title": "Page 1",
                "body": {"storage": {"value": "<p>Content 1</p>"}},
                "space": {"key": "DOCS"},
                "version": {"number": 1},
                "_links": {"webui": "/spaces/DOCS/pages/1"},
                "history": {"lastUpdated": {}},
            },
            {
                "id": "2",
                "title": "Page 2",
                "body": {"storage": {"value": "<p>Content 2</p>"}},
                "space": {"key": "DOCS"},
                "version": {"number": 2},
                "_links": {"webui": "/spaces/DOCS/pages/2"},
                "history": {"lastUpdated": {}},
            },
        ]
        mock_confluence_instance.get_all_pages_from_space.return_value = mock_pages

        mock_confluence_class = MagicMock(return_value=mock_confluence_instance)
        mock_atlassian = MagicMock()
        mock_atlassian.Confluence = mock_confluence_class

        mock_soup = MagicMock()
        mock_soup.get_text.side_effect = ["Content 1", "Content 2"]
        mock_bs4 = MagicMock()
        mock_bs4.BeautifulSoup.return_value = mock_soup

        with patch.dict("sys.modules", {"atlassian": mock_atlassian, "bs4": mock_bs4}):
            loader = ConfluenceLoader(
                url="https://test.atlassian.net",
                username="user@example.com",
                api_token="token",
                space_key="DOCS",
            )
            docs = loader.load()

        assert len(docs) == 2
        assert docs[0].text == "Content 1"
        assert docs[1].text == "Content 2"
        assert docs[0].metadata["title"] == "Page 1"
        assert docs[1].metadata["title"] == "Page 2"

    @pytest.mark.asyncio
    async def test_aload_space_pages(self):
        mock_confluence_instance = MagicMock()
        mock_pages = [
            {
                "id": "1",
                "title": "Async Page 1",
                "body": {"storage": {"value": "<p>Async Content 1</p>"}},
                "space": {"key": "DOCS"},
                "version": {"number": 1},
                "_links": {"webui": "/spaces/DOCS/pages/1"},
                "history": {"lastUpdated": {}},
            },
        ]
        mock_confluence_instance.get_all_pages_from_space.return_value = mock_pages

        mock_confluence_class = MagicMock(return_value=mock_confluence_instance)
        mock_atlassian = MagicMock()
        mock_atlassian.Confluence = mock_confluence_class

        mock_soup = MagicMock()
        mock_soup.get_text.return_value = "Async Content 1"
        mock_bs4 = MagicMock()
        mock_bs4.BeautifulSoup.return_value = mock_soup

        with patch.dict("sys.modules", {"atlassian": mock_atlassian, "bs4": mock_bs4}):
            loader = ConfluenceLoader(
                url="https://test.atlassian.net",
                username="user@example.com",
                api_token="token",
                space_key="DOCS",
            )
            docs = await loader.aload()

        assert len(docs) == 1
        assert docs[0].text == "Async Content 1"

    def test_pagination(self):
        mock_confluence_instance = MagicMock()

        first_batch = [
            {
                "id": str(i),
                "title": f"Page {i}",
                "body": {"storage": {"value": f"<p>Content {i}</p>"}},
                "space": {"key": "DOCS"},
                "version": {"number": 1},
                "_links": {"webui": f"/pages/{i}"},
                "history": {"lastUpdated": {}},
            }
            for i in range(100)
        ]
        second_batch = [
            {
                "id": "100",
                "title": "Page 100",
                "body": {"storage": {"value": "<p>Last page</p>"}},
                "space": {"key": "DOCS"},
                "version": {"number": 1},
                "_links": {"webui": "/pages/100"},
                "history": {"lastUpdated": {}},
            }
        ]

        mock_confluence_instance.get_all_pages_from_space.side_effect = [
            first_batch,
            second_batch,
        ]

        mock_confluence_class = MagicMock(return_value=mock_confluence_instance)
        mock_atlassian = MagicMock()
        mock_atlassian.Confluence = mock_confluence_class

        mock_soup = MagicMock()
        mock_soup.get_text.side_effect = [f"Content {i}" for i in range(101)]
        mock_bs4 = MagicMock()
        mock_bs4.BeautifulSoup.return_value = mock_soup

        with patch.dict("sys.modules", {"atlassian": mock_atlassian, "bs4": mock_bs4}):
            loader = ConfluenceLoader(
                url="https://test.atlassian.net",
                username="user@example.com",
                api_token="token",
                space_key="DOCS",
            )
            docs = loader.load()

        assert len(docs) == 101
        assert mock_confluence_instance.get_all_pages_from_space.call_count == 2

    def test_empty_space(self):
        mock_confluence_instance = MagicMock()
        mock_confluence_instance.get_all_pages_from_space.return_value = []

        mock_confluence_class = MagicMock(return_value=mock_confluence_instance)
        mock_atlassian = MagicMock()
        mock_atlassian.Confluence = mock_confluence_class

        mock_bs4 = MagicMock()

        with patch.dict("sys.modules", {"atlassian": mock_atlassian, "bs4": mock_bs4}):
            loader = ConfluenceLoader(
                url="https://test.atlassian.net",
                username="user@example.com",
                api_token="token",
                space_key="EMPTY",
            )
            docs = loader.load()

        assert docs == []

    def test_limit_parameter(self):
        mock_confluence_instance = MagicMock()
        mock_pages = [
            {
                "id": str(i),
                "title": f"Page {i}",
                "body": {"storage": {"value": f"<p>Content {i}</p>"}},
                "space": {"key": "DOCS"},
                "version": {"number": 1},
                "_links": {"webui": f"/pages/{i}"},
                "history": {"lastUpdated": {}},
            }
            for i in range(150)
        ]
        mock_confluence_instance.get_all_pages_from_space.return_value = mock_pages

        mock_confluence_class = MagicMock(return_value=mock_confluence_instance)
        mock_atlassian = MagicMock()
        mock_atlassian.Confluence = mock_confluence_class

        mock_soup = MagicMock()
        mock_soup.get_text.side_effect = [f"Content {i}" for i in range(10)]
        mock_bs4 = MagicMock()
        mock_bs4.BeautifulSoup.return_value = mock_soup

        with patch.dict("sys.modules", {"atlassian": mock_atlassian, "bs4": mock_bs4}):
            loader = ConfluenceLoader(
                url="https://test.atlassian.net",
                username="user@example.com",
                api_token="token",
                space_key="DOCS",
                limit=10,
            )
            docs = loader.load()

        assert len(docs) == 10

    def test_html_to_text_conversion(self):
        pytest.importorskip("bs4")

        mock_confluence_instance = MagicMock()
        mock_page = {
            "id": "1",
            "title": "Rich Content",
            "body": {
                "storage": {
                    "value": """
                    <h1>Heading</h1>
                    <p>Paragraph with <strong>bold</strong> and <em>italic</em></p>
                    <ul>
                        <li>Item 1</li>
                        <li>Item 2</li>
                    </ul>
                    <a href="http://example.com">Link</a>
                    """
                }
            },
            "space": {"key": "TEST"},
            "version": {"number": 1},
            "_links": {"webui": "/pages/1"},
            "history": {"lastUpdated": {}},
        }
        mock_confluence_instance.get_page_by_id.return_value = mock_page

        mock_confluence_class = MagicMock(return_value=mock_confluence_instance)
        mock_atlassian = MagicMock()
        mock_atlassian.Confluence = mock_confluence_class

        with patch.dict("sys.modules", {"atlassian": mock_atlassian}):
            loader = ConfluenceLoader(
                url="https://test.atlassian.net",
                username="user@example.com",
                api_token="token",
                page_id="1",
            )
            docs = loader.load()

        text = docs[0].text
        assert "Heading" in text
        assert "Paragraph" in text
        assert "Item 1" in text
        assert "Link" in text
        assert "<h1>" not in text
        assert "<p>" not in text

    def test_url_trailing_slash_handling(self):
        mock_confluence_instance = MagicMock()
        mock_page = {
            "id": "1",
            "title": "Test",
            "body": {"storage": {"value": "<p>Test</p>"}},
            "space": {"key": "TEST"},
            "version": {"number": 1},
            "_links": {"webui": "/pages/1"},
            "history": {"lastUpdated": {}},
        }
        mock_confluence_instance.get_page_by_id.return_value = mock_page

        mock_confluence_class = MagicMock(return_value=mock_confluence_instance)
        mock_atlassian = MagicMock()
        mock_atlassian.Confluence = mock_confluence_class

        mock_soup = MagicMock()
        mock_soup.get_text.return_value = "Test"
        mock_bs4 = MagicMock()
        mock_bs4.BeautifulSoup.return_value = mock_soup

        with patch.dict("sys.modules", {"atlassian": mock_atlassian, "bs4": mock_bs4}):
            loader = ConfluenceLoader(
                url="https://test.atlassian.net/",
                username="user@example.com",
                api_token="token",
                page_id="1",
            )
            docs = loader.load()

        assert "https://test.atlassian.net/pages/1" in docs[0].metadata["url"]
        assert "//" not in docs[0].metadata["url"].replace("https://", "")

    def test_metadata_extraction(self):
        mock_confluence_instance = MagicMock()
        mock_page = {
            "id": "999",
            "title": "Metadata Test",
            "body": {"storage": {"value": "<p>Content</p>"}},
            "space": {"key": "META"},
            "version": {"number": 5},
            "_links": {"webui": "/spaces/META/pages/999/Metadata+Test"},
            "history": {
                "lastUpdated": {
                    "by": {"displayName": "Jane Smith"},
                    "when": "2024-12-01T10:30:00Z",
                }
            },
        }
        mock_confluence_instance.get_page_by_id.return_value = mock_page

        mock_confluence_class = MagicMock(return_value=mock_confluence_instance)
        mock_atlassian = MagicMock()
        mock_atlassian.Confluence = mock_confluence_class

        mock_soup = MagicMock()
        mock_soup.get_text.return_value = "Content"
        mock_bs4 = MagicMock()
        mock_bs4.BeautifulSoup.return_value = mock_soup

        with patch.dict("sys.modules", {"atlassian": mock_atlassian, "bs4": mock_bs4}):
            loader = ConfluenceLoader(
                url="https://company.atlassian.net",
                username="user@example.com",
                api_token="token",
                page_id="999",
            )
            docs = loader.load()

        metadata = docs[0].metadata
        assert metadata["source"] == "confluence"
        assert metadata["title"] == "Metadata Test"
        assert metadata["page_id"] == "999"
        assert metadata["space"] == "META"
        assert metadata["version"] == 5
        assert metadata["author"] == "Jane Smith"
        assert metadata["last_modified"] == "2024-12-01T10:30:00Z"
        assert "company.atlassian.net" in metadata["url"]
