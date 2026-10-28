"""GitHub API Tool: interact with the GitHub REST API."""

from __future__ import annotations

import os
from typing import Any

from ..base import BaseTool, ToolResult


class GitHubAPITool(BaseTool):
    """Interact with the GitHub REST API.

    Supports searching repos, getting repo info, searching issues, getting
    issue details, and fetching pull requests together with their unified
    diff.  Uses stdlib ``urllib`` — no extra dependencies.

    Usage::

        tool = GitHubAPITool(token="ghp_...")
        result = await tool.run(action="search_repos", query="langchain")
        pr = await tool.run(
            action="get_pull_request_diff",
            owner="SynapseKit",
            repo="SynapseKit",
            pull_number=1056,
        )

    ``base_url`` defaults to the public API host and can be overridden to point
    at a GitHub Enterprise instance (or a test server).
    """

    name = "github_api"
    description = (
        "Interact with GitHub API to search repos, get repo info, "
        "search issues, get issue details, get a pull request, and "
        "fetch a pull request's unified diff."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "Action to perform",
                "enum": [
                    "search_repos",
                    "get_repo",
                    "search_issues",
                    "get_issue",
                    "get_pull_request",
                    "get_pull_request_diff",
                ],
            },
            "query": {
                "type": "string",
                "description": "Search query (for search_repos and search_issues)",
            },
            "owner": {
                "type": "string",
                "description": (
                    "Repository owner (for get_repo, get_issue, get_pull_request, "
                    "get_pull_request_diff)"
                ),
            },
            "repo": {
                "type": "string",
                "description": (
                    "Repository name (for get_repo, get_issue, get_pull_request, "
                    "get_pull_request_diff)"
                ),
            },
            "issue_number": {
                "type": "integer",
                "description": "Issue number (for get_issue)",
            },
            "pull_number": {
                "type": "integer",
                "description": (
                    "Pull request number (for get_pull_request and get_pull_request_diff)"
                ),
            },
        },
        "required": ["action"],
    }

    def __init__(self, token: str | None = None, base_url: str = "https://api.github.com") -> None:
        self._token = token or os.environ.get("GITHUB_TOKEN")
        self._base_url = base_url.rstrip("/")

    async def run(self, action: str = "", **kwargs: Any) -> ToolResult:
        if not action:
            return ToolResult(output="", error="No action specified.")

        handlers = {
            "search_repos": self._search_repos,
            "get_repo": self._get_repo,
            "search_issues": self._search_issues,
            "get_issue": self._get_issue,
            "get_pull_request": self._get_pull_request,
            "get_pull_request_diff": self._get_pull_request_diff,
        }

        handler = handlers.get(action)
        if handler is None:
            return ToolResult(
                output="",
                error=f"Unknown action: {action}. Must be one of: {', '.join(handlers)}",
            )

        try:
            return await handler(**kwargs)
        except Exception as e:
            return ToolResult(output="", error=f"GitHub API error: {e}")

    def _build_request(self, url: str, accept: str = "application/vnd.github.v3+json") -> Any:
        import urllib.request

        headers = {
            "User-Agent": "SynapseKit/1.0",
            "Accept": accept,
        }
        if self._token:
            headers["Authorization"] = f"token {self._token}"
        return urllib.request.Request(url, headers=headers)

    async def _api_get(self, url: str) -> Any:
        import asyncio
        import json
        import urllib.request

        loop = asyncio.get_event_loop()
        req = self._build_request(url)

        def _fetch():
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())

        return await loop.run_in_executor(None, _fetch)

    async def _api_get_text(self, url: str, accept: str) -> str:
        import asyncio
        import urllib.request

        loop = asyncio.get_running_loop()
        req = self._build_request(url, accept=accept)

        def _fetch() -> str:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.read().decode()

        return await loop.run_in_executor(None, _fetch)

    async def _search_repos(self, query: str = "", **kwargs: Any) -> ToolResult:
        if not query:
            return ToolResult(output="", error="No query provided for search_repos.")

        from urllib.parse import quote_plus

        data = await self._api_get(
            f"{self._base_url}/search/repositories?q={quote_plus(query)}&per_page=5"
        )
        items = data.get("items", [])
        if not items:
            return ToolResult(output="No repositories found.")

        results = []
        for i, repo in enumerate(items, 1):
            results.append(
                f"{i}. **{repo['full_name']}** ({repo.get('stargazers_count', 0)} stars)\n"
                f"   {repo.get('description', 'No description')}\n"
                f"   URL: {repo.get('html_url', '')}"
            )
        return ToolResult(output="\n\n".join(results))

    async def _get_repo(self, owner: str = "", repo: str = "", **kwargs: Any) -> ToolResult:
        if not owner or not repo:
            return ToolResult(output="", error="Both 'owner' and 'repo' are required.")

        data = await self._api_get(f"{self._base_url}/repos/{owner}/{repo}")
        return ToolResult(
            output=(
                f"**{data['full_name']}** ({data.get('stargazers_count', 0)} stars, "
                f"{data.get('forks_count', 0)} forks)\n"
                f"Description: {data.get('description', 'None')}\n"
                f"Language: {data.get('language', 'N/A')}\n"
                f"URL: {data.get('html_url', '')}"
            )
        )

    async def _search_issues(self, query: str = "", **kwargs: Any) -> ToolResult:
        if not query:
            return ToolResult(output="", error="No query provided for search_issues.")

        from urllib.parse import quote_plus

        data = await self._api_get(
            f"{self._base_url}/search/issues?q={quote_plus(query)}&per_page=5"
        )
        items = data.get("items", [])
        if not items:
            return ToolResult(output="No issues found.")

        results = []
        for i, issue in enumerate(items, 1):
            results.append(
                f"{i}. **{issue['title']}** (#{issue['number']})\n"
                f"   State: {issue.get('state', 'unknown')}\n"
                f"   URL: {issue.get('html_url', '')}"
            )
        return ToolResult(output="\n\n".join(results))

    async def _get_issue(
        self, owner: str = "", repo: str = "", issue_number: int = 0, **kwargs: Any
    ) -> ToolResult:
        if not owner or not repo or not issue_number:
            return ToolResult(output="", error="'owner', 'repo', and 'issue_number' are required.")

        data = await self._api_get(f"{self._base_url}/repos/{owner}/{repo}/issues/{issue_number}")
        body = (data.get("body") or "No description")[:500]
        return ToolResult(
            output=(
                f"**{data['title']}** (#{data['number']})\n"
                f"State: {data.get('state', 'unknown')}\n"
                f"Author: {data.get('user', {}).get('login', 'unknown')}\n"
                f"URL: {data.get('html_url', '')}\n\n"
                f"{body}"
            )
        )

    async def _get_pull_request(
        self, owner: str = "", repo: str = "", pull_number: int = 0, **kwargs: Any
    ) -> ToolResult:
        if not owner or not repo or not pull_number:
            return ToolResult(output="", error="'owner', 'repo', and 'pull_number' are required.")

        data = await self._api_get(f"{self._base_url}/repos/{owner}/{repo}/pulls/{pull_number}")
        body = (data.get("body") or "No description")[:500]
        return ToolResult(
            output=(
                f"**{data['title']}** (#{data['number']})\n"
                f"State: {data.get('state', 'unknown')}\n"
                f"Author: {data.get('user', {}).get('login', 'unknown')}\n"
                f"Changes: +{data.get('additions', 0)} -{data.get('deletions', 0)} "
                f"across {data.get('changed_files', 0)} file(s)\n"
                f"URL: {data.get('html_url', '')}\n\n"
                f"{body}"
            )
        )

    async def _get_pull_request_diff(
        self, owner: str = "", repo: str = "", pull_number: int = 0, **kwargs: Any
    ) -> ToolResult:
        if not owner or not repo or not pull_number:
            return ToolResult(output="", error="'owner', 'repo', and 'pull_number' are required.")

        diff = await self._api_get_text(
            f"{self._base_url}/repos/{owner}/{repo}/pulls/{pull_number}",
            accept="application/vnd.github.v3.diff",
        )
        if not diff.strip():
            return ToolResult(output="(empty diff)")
        return ToolResult(output=diff)
