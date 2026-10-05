import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from synapsekit.agents.tools.gitlab_api import GitLabAPITool

PROJECTS_SEARCH_RESPONSE = [
    {
        "path_with_namespace": "test/repo",
        "star_count": 10,
        "description": "Test",
        "web_url": "https://gitlab.example.com/test/repo",
    }
]

PROJECT_RESPONSE = {
    "path_with_namespace": "owner/repo",
    "star_count": 5,
    "forks_count": 2,
    "description": "Desc",
    "web_url": "https://gitlab.example.com/owner/repo",
}

ISSUES_SEARCH_RESPONSE = [
    {
        "title": "Something is broken",
        "iid": 42,
        "state": "opened",
        "web_url": "https://gitlab.example.com/owner/repo/-/issues/42",
    }
]

ISSUE_RESPONSE = {
    "title": "Something is broken",
    "iid": 42,
    "state": "opened",
    "author": {"username": "octocat"},
    "web_url": "https://gitlab.example.com/owner/repo/-/issues/42",
    "description": "Steps to reproduce...",
}

ROUTES = {
    "/api/v4/projects": PROJECTS_SEARCH_RESPONSE,
    "/api/v4/projects/owner%2Frepo": PROJECT_RESPONSE,
    "/api/v4/issues": ISSUES_SEARCH_RESPONSE,
    "/api/v4/projects/owner%2Frepo/issues/42": ISSUE_RESPONSE,
}


class _FakeGitLabHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        query = self.path[len(path) :]

        if path == "/api/v4/projects" and "search=empty" in query:
            body: object = []
        elif path in ROUTES:
            body = ROUTES[path]
        elif path == "/api/v4/projects/missing%2Frepo":
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"message": "404 Project Not Found"}')
            return
        else:
            self.send_response(404)
            self.end_headers()
            return

        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def gitlab_server() -> Iterator[str]:
    server = HTTPServer(("127.0.0.1", 0), _FakeGitLabHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()


@pytest.mark.asyncio
async def test_gitlab_search_projects(gitlab_server: str) -> None:
    tool = GitLabAPITool(token="test", url=gitlab_server)
    result = await tool.run(action="search_projects", query="test")

    assert result.error is None
    assert "**test/repo**" in result.output
    assert "https://gitlab.example.com/test/repo" in result.output


@pytest.mark.asyncio
async def test_gitlab_search_projects_no_results(gitlab_server: str) -> None:
    tool = GitLabAPITool(token="test", url=gitlab_server)
    result = await tool.run(action="search_projects", query="empty")

    assert result.error is None
    assert result.output == "No projects found."


@pytest.mark.asyncio
async def test_gitlab_get_project(gitlab_server: str) -> None:
    tool = GitLabAPITool(token="test", url=gitlab_server)
    result = await tool.run(action="get_project", owner="owner", repo="repo")

    assert result.error is None
    assert "**owner/repo**" in result.output
    assert "Desc" in result.output


@pytest.mark.asyncio
async def test_gitlab_get_project_not_found(gitlab_server: str) -> None:
    tool = GitLabAPITool(token="test", url=gitlab_server)
    result = await tool.run(action="get_project", owner="missing", repo="repo")

    assert result.error is not None
    assert "GitLab API error" in result.error


@pytest.mark.asyncio
async def test_gitlab_search_issues(gitlab_server: str) -> None:
    tool = GitLabAPITool(token="test", url=gitlab_server)
    result = await tool.run(action="search_issues", query="broken")

    assert result.error is None
    assert "Something is broken" in result.output
    assert "#42" in result.output


@pytest.mark.asyncio
async def test_gitlab_get_issue(gitlab_server: str) -> None:
    tool = GitLabAPITool(token="test", url=gitlab_server)
    result = await tool.run(action="get_issue", owner="owner", repo="repo", issue_number=42)

    assert result.error is None
    assert "Something is broken" in result.output
    assert "octocat" in result.output
    assert "Steps to reproduce" in result.output


@pytest.mark.asyncio
async def test_gitlab_missing_action() -> None:
    tool = GitLabAPITool(token="test")
    result = await tool.run(action="")
    assert result.error is not None
    assert "No action specified" in result.error


@pytest.mark.asyncio
async def test_gitlab_invalid_action() -> None:
    tool = GitLabAPITool(token="test")
    result = await tool.run(action="invalid_action")
    assert result.error is not None
    assert "Unknown action: invalid_action" in result.error


@pytest.mark.asyncio
async def test_gitlab_search_projects_requires_query(gitlab_server: str) -> None:
    tool = GitLabAPITool(token="test", url=gitlab_server)
    result = await tool.run(action="search_projects")
    assert result.error is not None
    assert "No query provided" in result.error


@pytest.mark.asyncio
async def test_gitlab_get_issue_requires_all_params(gitlab_server: str) -> None:
    tool = GitLabAPITool(token="test", url=gitlab_server)
    result = await tool.run(action="get_issue", owner="owner", repo="repo")
    assert result.error is not None
    assert "'owner', 'repo', and 'issue_number' are required" in result.error
