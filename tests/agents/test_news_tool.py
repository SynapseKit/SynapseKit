from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest

from synapsekit.agents.tools.news import NewsTool


def _make_tool(**kw):
    defaults = {"api_key": "test-key-123"}
    defaults.update(kw)
    return NewsTool(**defaults)


def _mock_response(data: dict):
    resp = MagicMock()
    resp.read.return_value = json.dumps(data).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


_HEADLINES_RESP = {
    "status": "ok",
    "totalResults": 2,
    "articles": [
        {
            "source": {"id": "bbc", "name": "BBC"},
            "title": "Test headline 1",
            "description": "Desc 1",
            "url": "https://example.com/1",
            "publishedAt": "2026-04-01T10:00:00Z",
        },
        {
            "source": {"id": None, "name": "CNN"},
            "title": "Test headline 2",
            "description": "Desc 2",
            "url": "https://example.com/2",
            "publishedAt": "2026-04-01T11:00:00Z",
        },
    ],
}

_SEARCH_RESP = {
    "status": "ok",
    "totalResults": 1,
    "articles": [
        {
            "source": {"id": "reuters", "name": "Reuters"},
            "title": "Search result",
            "description": "Found article",
            "url": "https://example.com/3",
            "publishedAt": "2026-03-30T08:00:00Z",
        },
    ],
}


class TestNewsTool:
    @pytest.mark.asyncio
    async def test_get_headlines(self):
        tool = _make_tool()
        with patch("urllib.request.urlopen", return_value=_mock_response(_HEADLINES_RESP)):
            res = await tool.run(action="get_headlines")
        assert not res.is_error
        articles = json.loads(res.output)
        assert len(articles) == 2
        assert articles[0]["title"] == "Test headline 1"
        assert articles[0]["source"] == "BBC"

    @pytest.mark.asyncio
    async def test_get_headlines_with_category(self):
        tool = _make_tool()
        with patch("urllib.request.urlopen", return_value=_mock_response(_HEADLINES_RESP)) as mock_open:
            res = await tool.run(action="get_headlines", country="gb", category="technology")
        assert not res.is_error
        url = mock_open.call_args[0][0].full_url
        assert "country=gb" in url
        assert "category=technology" in url

    @pytest.mark.asyncio
    async def test_search(self):
        tool = _make_tool()
        with patch("urllib.request.urlopen", return_value=_mock_response(_SEARCH_RESP)):
            res = await tool.run(action="search", query="python")
        assert not res.is_error
        articles = json.loads(res.output)
        assert len(articles) == 1
        assert articles[0]["title"] == "Search result"

    @pytest.mark.asyncio
    async def test_search_with_params(self):
        tool = _make_tool()
        with patch("urllib.request.urlopen", return_value=_mock_response(_SEARCH_RESP)) as mock_open:
            res = await tool.run(
                action="search", query="ai", from_date="2026-03-01", sort_by="popularity"
            )
        assert not res.is_error
        url = mock_open.call_args[0][0].full_url
        assert "q=ai" in url
        assert "from=2026-03-01" in url
        assert "sortBy=popularity" in url

    @pytest.mark.asyncio
    async def test_search_missing_query(self):
        tool = _make_tool()
        res = await tool.run(action="search")
        assert res.is_error
        assert "query" in res.error.lower()

    @pytest.mark.asyncio
    async def test_no_action(self):
        tool = _make_tool()
        res = await tool.run()
        assert res.is_error
        assert "No action" in res.error

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        tool = _make_tool()
        res = await tool.run(action="delete")
        assert res.is_error
        assert "Unknown action" in res.error

    @pytest.mark.asyncio
    async def test_missing_api_key(self):
        tool = NewsTool(api_key="")
        res = await tool.run(action="get_headlines")
        assert res.is_error
        assert "NEWS_API_KEY" in res.error

    @pytest.mark.asyncio
    async def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("NEWS_API_KEY", "env-key-456")
        tool = NewsTool()
        assert tool._key == "env-key-456"

    @pytest.mark.asyncio
    async def test_api_error_response(self):
        err_resp = {"status": "error", "message": "rate limit exceeded"}
        tool = _make_tool()
        with patch("urllib.request.urlopen", return_value=_mock_response(err_resp)):
            res = await tool.run(action="get_headlines")
        assert res.is_error
        assert "rate limit" in res.error

    @pytest.mark.asyncio
    async def test_network_error(self):
        tool = _make_tool()
        with patch("urllib.request.urlopen", side_effect=Exception("timeout")):
            res = await tool.run(action="get_headlines")
        assert res.is_error
        assert "NewsAPI error" in res.error

    @pytest.mark.asyncio
    async def test_empty_results(self):
        empty = {"status": "ok", "totalResults": 0, "articles": []}
        tool = _make_tool()
        with patch("urllib.request.urlopen", return_value=_mock_response(empty)):
            res = await tool.run(action="get_headlines")
        assert not res.is_error
        assert "No articles" in res.output

    def test_schema(self):
        tool = _make_tool()
        s = tool.schema()
        assert s["function"]["name"] == "news"
        props = s["function"]["parameters"]["properties"]
        assert "action" in props
        assert "query" in props
        assert "country" in props
