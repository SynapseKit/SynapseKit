"""GitHub API Tool: interact with the GitHub REST API."""

from __future__ import annotations

import os
from typing import Any

from ..base import BaseTool, ToolResult


class GitHubAPITool(BaseTool):
    """Interact with the GitHub REST API.

    Supports searching repos, getting repo info, searching issues, and
    getting issue details.  Uses stdlib ``urllib`` — no extra dependencies.

    Usage::

        tool = GitHubAPITool(token="ghp_...")
        result = await tool.run(action="search_repos", query="langchain")
    """

    name = "github_api"
    description = (
        "Interact with GitHub API to search repos, get repo info, "
        "search issues, and get issue details."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "Action to perform",
                "enum": ["search_repos", "get_repo", "search_issues", "get_issue"],
            },
            "query": {
                "type": "string",
                "description": "Search query (for search_repos and search_issues)",
            },
            "owner": {
                "type": "string",
                "description": "Repository owner (for get_repo and get_issue)",
            },
            "repo": {
                "type": "string",
                "description": "Repository name (for get_repo and get_issue)",
            },
            "issue_number": {
                "type": "integer",
                "description": "Issue number (for get_issue)",
            },
        },
        "required": ["action"],
    }

    def __init__(self, token: str | None = None) -> None:
        self._token = token or os.environ.get("GITHUB_TOKEN")

    async def run(self, action: str = "", **kwargs: Any) -> ToolResult:
        if not action:
            return ToolResult(output="", error="No action specified.")

        handlers = {
            "search_repos": self._search_repos,
            "get_repo": self._get_repo,
            "search_issues": self._search_issues,
            "get_issue": self._get_issue,
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

    def _build_request(self, url: str) -> Any:
        import urllib.request

        headers = {
            "User-Agent": "SynapseKit/1.0",
            "Accept": "application/vnd.github.v3+json",
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

    async def _search_repos(self, query: str = "", **kwargs: Any) -> ToolResult:
        if not query:
            return ToolResult(output="", error="No query provided for search_repos.")

        from urllib.parse import quote_plus

        data = await self._api_get(
            f"https://api.github.com/search/repositories?q={quote_plus(query)}&per_page=5"
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

        data = await self._api_get(f"https://api.github.com/repos/{owner}/{repo}")
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
            f"https://api.github.com/search/issues?q={quote_plus(query)}&per_page=5"
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

        data = await self._api_get(
            f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
        )
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
