from unittest.mock import MagicMock, patch

import pytest

from synapsekit.agents.tools.google_search import GoogleSearchTool

try:
    import serpapi  # noqa: F401

    SERPAPI_AVAILABLE = True
except ImportError:
    SERPAPI_AVAILABLE = False


@pytest.fixture
def mock_serpapi():
    if not SERPAPI_AVAILABLE:
        yield None
        return
    with patch("serpapi.GoogleSearch") as mock_search_cls:
        mock_search_instance = MagicMock()
        mock_search_cls.return_value = mock_search_instance
        yield mock_search_instance


@pytest.mark.skipif(not SERPAPI_AVAILABLE, reason="serpapi not installed")
@pytest.mark.asyncio
async def test_google_search_success(mock_serpapi):
    mock_serpapi.get_dict.return_value = {
        "organic_results": [
            {
                "title": "SynapseKit",
                "link": "https://github.com/SynapseKit",
                "snippet": "A lightweight RAG framework.",
            },
            {"title": "Python", "link": "https://python.org", "snippet": "Programming language."},
        ]
    }

    tool = GoogleSearchTool(api_key="test_key")
    result = await tool.run(query="synapsekit python")

    assert result.error is None
    assert "**SynapseKit**" in result.output
    assert "https://github.com/SynapseKit" in result.output
    assert "**Python**" in result.output


@pytest.mark.asyncio
async def test_google_search_no_api_key():

    tool = GoogleSearchTool(api_key=None)
    with patch.dict("os.environ", clear=True):
        result = await tool.run(query="test")
        assert result.error is not None
        assert "SerpAPI key not provided" in result.error


@pytest.mark.asyncio
async def test_google_search_empty_query():
    tool = GoogleSearchTool(api_key="test")
    result = await tool.run(query="")
    assert result.error is not None
    assert "No search query provided" in result.error


@pytest.mark.skipif(not SERPAPI_AVAILABLE, reason="serpapi not installed")
@pytest.mark.asyncio
async def test_google_search_no_results(mock_serpapi):
    mock_serpapi.get_dict.return_value = {"organic_results": []}

    tool = GoogleSearchTool(api_key="test_key")
    result = await tool.run(query="adkjhkjasdhkjas")

    assert result.error is None
    assert "No results found" in result.output


@pytest.mark.skipif(not SERPAPI_AVAILABLE, reason="serpapi not installed")
@pytest.mark.asyncio
async def test_google_search_api_error(mock_serpapi):
    mock_serpapi.get_dict.return_value = {"error": "Invalid API key."}

    tool = GoogleSearchTool(api_key="bad_key")
    result = await tool.run(query="test")

    assert result.error is not None
    assert "SerpAPI Error: Invalid API key." in result.error
