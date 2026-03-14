from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.synapsekit.agents.tools.web_scraper import WebScraperTool


class TestWebScraperTool:
    def test_init(self):
        tool = WebScraperTool(timeout=60)
        assert tool.timeout == 60
        assert tool.name == "web_scraper"

    @pytest.mark.asyncio
    async def test_run_success(self):
        tool = WebScraperTool()
        mock_html = "<html><body><p>Hello World</p></body></html>"

        # Создаём моки для ответа и клиента
        mock_response = MagicMock()
        mock_response.text = mock_html
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tool.run(url="https://example.com")

            assert result.error is None
            assert "Hello World" in result.output

    @pytest.mark.asyncio
    async def test_run_with_css_selector(self):
        tool = WebScraperTool()
        mock_html = "<html><body><article><p>Article text</p></article><nav>Nav text</nav></body></html>"

        mock_response = MagicMock()
        mock_response.text = mock_html
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tool.run(url="https://example.com", css_selector="article")

            assert result.error is None
            assert "Article text" in result.output
            assert "Nav text" not in result.output

    def test_run_sync_success(self):
        tool = WebScraperTool()
        mock_html = "<html><body><h1>Test</h1></body></html>"

        mock_response = MagicMock()
        mock_response.text = mock_html
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get = MagicMock(return_value=mock_response)
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)

        with patch("httpx.Client", return_value=mock_client):
            result = tool.run_sync(url="https://example.com")

            assert result.error is None
            assert "Test" in result.output

    @pytest.mark.asyncio
    async def test_run_no_url(self):
        tool = WebScraperTool()
        result = await tool.run()
        assert result.error is not None
        assert "No URL provided" in result.error

    @pytest.mark.asyncio
    async def test_run_http_error(self):
        tool = WebScraperTool()

        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=Exception("Connection failed"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tool.run(url="https://example.com")

            assert result.error is not None
            assert "Scraping failed" in result.error
