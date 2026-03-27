"""Tests for BingSearchTool."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from synapsekit.agents.tools.bing_search import BingSearchTool


@pytest.fixture
def mock_response():
    return {
        "webPages": {
            "value": [
                {
                    "name": "Example Article",
                    "url": "https://example.com/article",
                    "snippet": "This is a snippet about AI",
                },
                {
                    "name": "Another Result",
                    "url": "https://example.com/another",
                    "snippet": "More information here",
                },
            ]
        }
    }


@pytest.fixture
def mock_empty_response():
    return {"webPages": {"value": []}}


@pytest.mark.asyncio
class TestBingSearchTool:
    async def test_requires_api_key(self):
        tool = BingSearchTool()
        result = await tool.run(query="test")
        assert result.error == "No BING_API_KEY configured."

    async def test_requires_query(self):
        tool = BingSearchTool(api_key="test-key")
        result = await tool.run()
        assert result.error == "No search query provided."

    @patch("httpx.AsyncClient.get")
    async def test_successful_search(self, mock_get, mock_response):
        mock_resp = Mock()
        mock_resp.json.return_value = mock_response
        mock_resp.raise_for_status = Mock()
        mock_get.return_value = mock_resp

        tool = BingSearchTool(api_key="test-key")
        result = await tool.run(query="AI news")

        assert result.error is None
        assert "Example Article" in result.output
        assert "https://example.com/article" in result.output

    @patch("httpx.AsyncClient.get")
    async def test_empty_results(self, mock_get, mock_empty_response):
        mock_resp = Mock()
        mock_resp.json.return_value = mock_empty_response
        mock_resp.raise_for_status = Mock()
        mock_get.return_value = mock_resp

        tool = BingSearchTool(api_key="test-key")
        result = await tool.run(query="nonexistent query")
        assert result.output == "No results found."

    @patch("httpx.AsyncClient.get")
    async def test_count_parameter(self, mock_get, mock_response):
        mock_resp = Mock()
        mock_resp.json.return_value = mock_response
        mock_resp.raise_for_status = Mock()
        mock_get.return_value = mock_resp

        tool = BingSearchTool(api_key="test-key")
        await tool.run(query="test", count=10)
        assert mock_get.call_args.kwargs["params"]["count"] == 10

    @patch("httpx.AsyncClient.get")
    async def test_market_parameter(self, mock_get, mock_response):
        mock_resp = Mock()
        mock_resp.json.return_value = mock_response
        mock_resp.raise_for_status = Mock()
        mock_get.return_value = mock_resp

        tool = BingSearchTool(api_key="test-key")
        await tool.run(query="test", market="en-GB")
        assert mock_get.call_args.kwargs["params"]["mkt"] == "en-GB"

    @patch("httpx.AsyncClient.get")
    async def test_max_count_limit(self, mock_get, mock_response):
        mock_resp = Mock()
        mock_resp.json.return_value = mock_response
        mock_resp.raise_for_status = Mock()
        mock_get.return_value = mock_resp

        tool = BingSearchTool(api_key="test-key")
        await tool.run(query="test", count=100)
        assert mock_get.call_args.kwargs["params"]["count"] == 50

    @patch("httpx.AsyncClient.get")
    async def test_http_error_handling(self, mock_get):
        import httpx

        mock_resp = Mock()
        mock_resp.status_code = 401
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Unauthorized", request=Mock(), response=mock_resp
        )
        mock_get.return_value = mock_resp

        tool = BingSearchTool(api_key="invalid-key")
        result = await tool.run(query="test")
        assert "Bing Search API error: 401" in result.error

    @patch("httpx.AsyncClient.get")
    async def test_generic_exception_handling(self, mock_get):
        mock_get.side_effect = Exception("Network error")

        tool = BingSearchTool(api_key="test-key")
        result = await tool.run(query="test")
        assert "Bing Search error:" in result.error

    async def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("BING_API_KEY", "env-key")
        tool = BingSearchTool()
        assert tool._api_key == "env-key"

    @patch("httpx.AsyncClient.get")
    async def test_snippet_truncation(self, mock_get):
        long_snippet = "A" * 500
        mock_resp = Mock()
        mock_resp.json.return_value = {
            "webPages": {
                "value": [
                    {
                        "name": "Test",
                        "url": "https://example.com",
                        "snippet": long_snippet,
                    }
                ]
            }
        }
        mock_resp.raise_for_status = Mock()
        mock_get.return_value = mock_resp

        tool = BingSearchTool(api_key="test-key")
        result = await tool.run(query="test")
        assert len(result.output.split("\n   ")[-1]) <= 310
