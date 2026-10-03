from unittest.mock import MagicMock, patch

import pytest

from synapsekit.agents.tools.gitlab_api import GitLabAPITool


@pytest.fixture
def mock_urlopen():
    with patch("urllib.request.urlopen") as mock_open:
        yield mock_open


@pytest.mark.asyncio
async def test_gitlab_search_projects(mock_urlopen):
    mock_resp = MagicMock()
    mock_resp.read.return_value = b'[{"path_with_namespace": "test/repo", "star_count": 10, "description": "Test", "web_url": "https://gitlab.com/test/repo"}]'
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    tool = GitLabAPITool(token="test")
    result = await tool.run(action="search_projects", query="test")

    assert result.error is None
    assert "**test/repo**" in result.output
    assert "https://gitlab.com/test/repo" in result.output


@pytest.mark.asyncio
async def test_gitlab_get_project(mock_urlopen):
    mock_resp = MagicMock()
    mock_resp.read.return_value = b'{"path_with_namespace": "owner/repo", "star_count": 5, "forks_count": 2, "description": "Desc", "web_url": "https://gitlab.com/owner/repo"}'
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    tool = GitLabAPITool(token="test")
    result = await tool.run(action="get_project", owner="owner", repo="repo")

    assert result.error is None
    assert "**owner/repo**" in result.output
    assert "Desc" in result.output


@pytest.mark.asyncio
async def test_gitlab_missing_action():
    tool = GitLabAPITool(token="test")
    result = await tool.run(action="")
    assert result.error is not None
    assert "No action specified" in result.error


@pytest.mark.asyncio
async def test_gitlab_invalid_action():
    tool = GitLabAPITool(token="test")
    result = await tool.run(action="invalid_action")
    assert result.error is not None
    assert "Unknown action: invalid_action" in result.error
