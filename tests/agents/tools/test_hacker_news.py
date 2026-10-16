from unittest.mock import MagicMock, patch

import pytest

from synapsekit.agents.tools.hacker_news import HackerNewsTool


@pytest.mark.asyncio
async def test_hacker_news_tool_top_stories():
    tool = HackerNewsTool()

    mock_top_stories = b"[1, 2, 3]"
    mock_story = b'{"title": "Test Story", "url": "http://example.com", "score": 100}'

    with patch("urllib.request.urlopen") as mock_urlopen:
        # First call gets the top stories list, next 3 calls get the individual stories
        mock_response_list = MagicMock()
        mock_response_list.read.return_value = mock_top_stories

        mock_response_story = MagicMock()
        mock_response_story.read.return_value = mock_story

        # Setup context manager returns
        mock_response_list.__enter__.return_value = mock_response_list
        mock_response_story.__enter__.return_value = mock_response_story

        mock_urlopen.side_effect = [
            mock_response_list,
            mock_response_story,
            mock_response_story,
            mock_response_story,
        ]

        result = await tool.run(action="top_stories")
        assert not result.error
        assert "Test Story" in result.output
        assert "Score: 100" in result.output


@pytest.mark.asyncio
async def test_hacker_news_tool_get_item():
    tool = HackerNewsTool()

    mock_story = b'{"id": 123, "title": "Specific Story", "type": "story"}'

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = mock_story
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        result = await tool.run(action="get_item", item_id=123)
        assert not result.error
        assert "Specific Story" in result.output
