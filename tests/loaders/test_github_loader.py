from __future__ import annotations

import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synapsekit.loaders.base import Document


def _make_mock_response(json_data=None, text_data=None, status_code=200):
    """Create a mock HTTP response."""
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_data
    response.text = text_data
    response.raise_for_status = MagicMock()
    if status_code >= 400:
        response.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return response


def test_import_error_without_httpx():
    from synapsekit.loaders.github import GitHubLoader

    with patch.dict("sys.modules", {"httpx": None}):
        loader = GitHubLoader("owner/repo")
        with pytest.raises(ImportError, match="httpx"):
            loader.load_sync()


def test_readme_loading():
    from synapsekit.loaders.github import GitHubLoader

    readme_content = "# Test Project\n\nThis is a test."
    encoded = base64.b64encode(readme_content.encode("utf-8")).decode("utf-8")
    
    mock_response = _make_mock_response(
        json_data={"content": encoded, "encoding": "base64"}
    )
    
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    
    with patch("httpx.AsyncClient", return_value=mock_client):
        loader = GitHubLoader("owner/repo", content_type="readme")
        docs = loader.load_sync()
    
    assert len(docs) == 1
    assert docs[0].text == readme_content
    assert docs[0].metadata["source"] == "github"
    assert docs[0].metadata["repo"] == "owner/repo"
    assert docs[0].metadata["type"] == "readme"
    assert "github.com/owner/repo" in docs[0].metadata["url"]


def test_issues_loading_filters_prs():
    from synapsekit.loaders.github import GitHubLoader

    issues_data = [
        {
            "number": 1,
            "title": "Bug report",
            "body": "There is a bug",
            "user": {"login": "user1"},
            "created_at": "2024-01-01T00:00:00Z",
            "html_url": "https://github.com/owner/repo/issues/1",
        },
        {
            "number": 2,
            "title": "Feature request",
            "body": "Add feature",
            "user": {"login": "user2"},
            "created_at": "2024-01-02T00:00:00Z",
            "html_url": "https://github.com/owner/repo/issues/2",
            "pull_request": {"url": "..."},  # This should be filtered out
        },
        {
            "number": 3,
            "title": "Another bug",
            "body": "Another issue",
            "user": {"login": "user3"},
            "created_at": "2024-01-03T00:00:00Z",
            "html_url": "https://github.com/owner/repo/issues/3",
        },
    ]
    
    mock_response = _make_mock_response(json_data=issues_data)
    
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    
    with patch("httpx.AsyncClient", return_value=mock_client):
        loader = GitHubLoader("owner/repo", content_type="issues")
        docs = loader.load_sync()
    
    # Should only have 2 docs (issue #2 filtered out as it's a PR)
    assert len(docs) == 2
    assert "Bug report" in docs[0].text
    assert "Another bug" in docs[1].text
    assert docs[0].metadata["type"] == "issue"
    assert docs[0].metadata["author"] == "user1"
    assert docs[0].metadata["number"] == 1


def test_prs_loading():
    from synapsekit.loaders.github import GitHubLoader

    prs_data = [
        {
            "number": 10,
            "title": "Add feature X",
            "body": "This PR adds X",
            "user": {"login": "contributor1"},
            "created_at": "2024-01-10T00:00:00Z",
            "html_url": "https://github.com/owner/repo/pull/10",
        },
        {
            "number": 11,
            "title": "Fix bug Y",
            "body": "This fixes Y",
            "user": {"login": "contributor2"},
            "created_at": "2024-01-11T00:00:00Z",
            "html_url": "https://github.com/owner/repo/pull/11",
        },
    ]
    
    mock_response = _make_mock_response(json_data=prs_data)
    
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    
    with patch("httpx.AsyncClient", return_value=mock_client):
        loader = GitHubLoader("owner/repo", content_type="prs")
        docs = loader.load_sync()
    
    assert len(docs) == 2
    assert "Add feature X" in docs[0].text
    assert docs[0].metadata["type"] == "pr"
    assert docs[0].metadata["author"] == "contributor1"
    assert docs[0].metadata["number"] == 10


def test_files_loading():
    from synapsekit.loaders.github import GitHubLoader

    repo_data = {"default_branch": "main"}
    tree_data = {
        "tree": [
            {"path": "README.md", "type": "blob"},
            {"path": "src/main.py", "type": "blob"},
            {"path": "src/", "type": "tree"},  # Directory, should be filtered
            {"path": "tests/test_main.py", "type": "blob"},
        ]
    }
    
    file_contents = {
        "README.md": "# Project",
        "src/main.py": "def main(): pass",
        "tests/test_main.py": "def test_main(): pass",
    }
    
    async def mock_get(url, headers=None):
        if "/repos/owner/repo/git/trees/" in url:
            return _make_mock_response(json_data=tree_data)
        elif url.startswith("https://api.github.com/repos/owner/repo"):
            return _make_mock_response(json_data=repo_data)
        elif "raw.githubusercontent.com" in url:
            # Extract filename from URL
            for filename, content in file_contents.items():
                if filename in url:
                    return _make_mock_response(text_data=content)
        return _make_mock_response(text_data="")
    
    mock_client = MagicMock()
    mock_client.get = AsyncMock(side_effect=mock_get)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    
    with patch("httpx.AsyncClient", return_value=mock_client):
        loader = GitHubLoader("owner/repo", content_type="files")
        docs = loader.load_sync()
    
    # Should have 3 files (tree directory filtered out)
    assert len(docs) == 3
    assert docs[0].metadata["type"] == "file"
    assert "path" in docs[0].metadata
    assert docs[0].metadata["repo"] == "owner/repo"


def test_files_with_path_filter():
    from synapsekit.loaders.github import GitHubLoader

    repo_data = {"default_branch": "main"}
    tree_data = {
        "tree": [
            {"path": "README.md", "type": "blob"},
            {"path": "src/main.py", "type": "blob"},
            {"path": "src/utils.py", "type": "blob"},
            {"path": "tests/test_main.py", "type": "blob"},
        ]
    }
    
    async def mock_get(url, headers=None):
        if "/repos/owner/repo/git/trees/" in url:
            return _make_mock_response(json_data=tree_data)
        elif url.startswith("https://api.github.com/repos/owner/repo"):
            return _make_mock_response(json_data=repo_data)
        elif "raw.githubusercontent.com" in url:
            if "src/" in url:
                return _make_mock_response(text_data="# Python code")
        return _make_mock_response(text_data="")
    
    mock_client = MagicMock()
    mock_client.get = AsyncMock(side_effect=mock_get)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    
    with patch("httpx.AsyncClient", return_value=mock_client):
        loader = GitHubLoader("owner/repo", content_type="files", path="src/")
        docs = loader.load_sync()
    
    # Should only have src/ files
    assert len(docs) == 2
    assert all("src/" in doc.metadata["path"] for doc in docs)


def test_limit_parameter():
    from synapsekit.loaders.github import GitHubLoader

    issues_data = [
        {
            "number": i,
            "title": f"Issue {i}",
            "body": f"Body {i}",
            "user": {"login": "user"},
            "created_at": "2024-01-01T00:00:00Z",
            "html_url": f"https://github.com/owner/repo/issues/{i}",
        }
        for i in range(10)
    ]
    
    mock_response = _make_mock_response(json_data=issues_data)
    
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    
    with patch("httpx.AsyncClient", return_value=mock_client):
        loader = GitHubLoader("owner/repo", content_type="issues", limit=3)
        docs = loader.load_sync()
    
    assert len(docs) == 3


def test_token_in_headers():
    from synapsekit.loaders.github import GitHubLoader

    readme_content = "# Test"
    encoded = base64.b64encode(readme_content.encode("utf-8")).decode("utf-8")
    mock_response = _make_mock_response(json_data={"content": encoded})
    
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    
    with patch("httpx.AsyncClient", return_value=mock_client):
        loader = GitHubLoader("owner/repo", token="test_token")
        loader.load_sync()
    
    # Check that token was passed in headers
    call_args = mock_client.get.call_args
    headers = call_args.kwargs.get("headers", {})
    assert headers.get("Authorization") == "Bearer test_token"


def test_retry_on_rate_limit():
    from synapsekit.loaders.github import GitHubLoader

    readme_content = "# Test"
    encoded = base64.b64encode(readme_content.encode("utf-8")).decode("utf-8")
    
    # First call returns 429, second call succeeds
    rate_limit_response = _make_mock_response(status_code=429)
    success_response = _make_mock_response(json_data={"content": encoded})
    
    mock_client = MagicMock()
    mock_client.get = AsyncMock(side_effect=[rate_limit_response, success_response])
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    
    with patch("httpx.AsyncClient", return_value=mock_client):
        with patch("asyncio.sleep", new_callable=AsyncMock):
            loader = GitHubLoader("owner/repo", content_type="readme")
            docs = loader.load_sync()
    
    # Should succeed after retry
    assert len(docs) == 1
    assert mock_client.get.call_count == 2


def test_invalid_content_type():
    from synapsekit.loaders.github import GitHubLoader

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    
    with patch("httpx.AsyncClient", return_value=mock_client):
        loader = GitHubLoader("owner/repo", content_type="invalid")  # type: ignore
        with pytest.raises(ValueError, match="Unknown content_type"):
            loader.load_sync()


def test_async_load():
    from synapsekit.loaders.github import GitHubLoader

    readme_content = "# Async Test"
    encoded = base64.b64encode(readme_content.encode("utf-8")).decode("utf-8")
    mock_response = _make_mock_response(json_data={"content": encoded})
    
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    
    async def run_test():
        with patch("httpx.AsyncClient", return_value=mock_client):
            loader = GitHubLoader("owner/repo", content_type="readme")
            docs = await loader.load()
        return docs
    
    docs = asyncio.run(run_test())
    assert len(docs) == 1
    assert docs[0].text == readme_content
