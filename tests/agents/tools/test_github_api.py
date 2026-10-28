import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from synapsekit.agents.tools.github_api import GitHubAPITool

_PR_JSON = {
    "number": 1056,
    "title": "Add widget",
    "state": "open",
    "user": {"login": "octocat"},
    "additions": 10,
    "deletions": 2,
    "changed_files": 1,
    "html_url": "https://github.com/acme/widgets/pull/1056",
    "body": "This PR adds a widget.",
}

_PR_DIFF = (
    "diff --git a/widget.py b/widget.py\n"
    "index 000..111 100644\n"
    "--- a/widget.py\n"
    "+++ b/widget.py\n"
    "@@ -1 +1,2 @@\n"
    " existing\n"
    "+new line\n"
)


class _FakeGitHubHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path != "/repos/acme/widgets/pulls/1056":
            self.send_response(404)
            self.end_headers()
            return

        # The diff variant is selected via the Accept header, exactly like the
        # real API — this is what the tool relies on.
        accept = self.headers.get("Accept", "")
        if "diff" in accept:
            payload = _PR_DIFF.encode()
            content_type = "application/vnd.github.v3.diff"
        else:
            payload = json.dumps(_PR_JSON).encode()
            content_type = "application/json"

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def github_base_url() -> Iterator[str]:
    server = HTTPServer(("127.0.0.1", 0), _FakeGitHubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()


@pytest.mark.asyncio
async def test_get_pull_request(github_base_url: str) -> None:
    tool = GitHubAPITool(base_url=github_base_url)
    result = await tool.run(
        action="get_pull_request", owner="acme", repo="widgets", pull_number=1056
    )

    assert not result.error
    assert "Add widget" in result.output
    assert "#1056" in result.output
    assert "+10 -2" in result.output
    assert "octocat" in result.output


@pytest.mark.asyncio
async def test_get_pull_request_diff(github_base_url: str) -> None:
    tool = GitHubAPITool(base_url=github_base_url)
    result = await tool.run(
        action="get_pull_request_diff", owner="acme", repo="widgets", pull_number=1056
    )

    assert not result.error
    assert "diff --git a/widget.py b/widget.py" in result.output
    assert "+new line" in result.output


@pytest.mark.asyncio
async def test_get_pull_request_requires_all_args(github_base_url: str) -> None:
    tool = GitHubAPITool(base_url=github_base_url)
    result = await tool.run(action="get_pull_request", owner="acme", repo="widgets")

    assert result.error
    assert "pull_number" in result.error


@pytest.mark.asyncio
async def test_get_pull_request_diff_requires_all_args(github_base_url: str) -> None:
    tool = GitHubAPITool(base_url=github_base_url)
    result = await tool.run(action="get_pull_request_diff", owner="acme", pull_number=1056)

    assert result.error
    assert "'owner', 'repo', and 'pull_number' are required." in result.error


@pytest.mark.asyncio
async def test_unknown_action() -> None:
    tool = GitHubAPITool(base_url="http://127.0.0.1:1")
    result = await tool.run(action="get_pull_request_comments")

    assert result.error
    assert "Unknown action" in result.error


def test_pull_request_actions_in_schema() -> None:
    enum = GitHubAPITool.parameters["properties"]["action"]["enum"]
    assert "get_pull_request" in enum
    assert "get_pull_request_diff" in enum
