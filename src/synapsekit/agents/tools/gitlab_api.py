"""GitLab API Tool: interact with the GitLab REST API."""

from __future__ import annotations

import os
from typing import Any

from ..base import BaseTool, ToolResult


class GitLabAPITool(BaseTool):
    """Interact with the GitLab REST API.

    Supports searching projects, getting project info, searching issues, and
    getting issue details. Uses stdlib ``urllib`` — no extra dependencies.

    Usage::

        tool = GitLabAPITool(token="glpat-...", url="https://gitlab.com")
        result = await tool.run(action="search_projects", query="synapsekit")
    """

    name = "gitlab_api"
    description = (
        "Interact with GitLab API to search projects, get project info, "
        "search issues, and get issue details."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "Action to perform",
                "enum": ["search_projects", "get_project", "search_issues", "get_issue"],
            },
            "query": {
                "type": "string",
                "description": "Search query (for search_projects and search_issues)",
            },
            "owner": {
                "type": "string",
                "description": "Project owner or group (for get_project and get_issue)",
            },
            "repo": {
                "type": "string",
                "description": "Project name (for get_project and get_issue)",
            },
            "issue_number": {
                "type": "integer",
                "description": "Issue number (for get_issue)",
            },
        },
        "required": ["action"],
    }

    def __init__(self, token: str | None = None, url: str | None = None) -> None:
        self._token = token or os.environ.get("GITLAB_TOKEN")
        self._url = (url or os.environ.get("GITLAB_URL") or "https://gitlab.com").rstrip("/")

    async def run(self, action: str = "", **kwargs: Any) -> ToolResult:
        if not action:
            return ToolResult(output="", error="No action specified.")

        handlers = {
            "search_projects": self._search_projects,
            "get_project": self._get_project,
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
            return ToolResult(output="", error=f"GitLab API error: {e}")

    def _build_request(self, url: str) -> Any:
        import urllib.request

        headers = {
            "User-Agent": "SynapseKit/1.0",
            "Accept": "application/json",
        }
        if self._token:
            headers["PRIVATE-TOKEN"] = self._token
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

    async def _search_projects(self, query: str = "", **kwargs: Any) -> ToolResult:
        if not query:
            return ToolResult(output="", error="No query provided for search_projects.")

        from urllib.parse import quote_plus

        data = await self._api_get(
            f"{self._url}/api/v4/projects?search={quote_plus(query)}&per_page=5"
        )
        if not data:
            return ToolResult(output="No projects found.")

        results = []
        for i, project in enumerate(data, 1):
            results.append(
                f"{i}. **{project['path_with_namespace']}** ({project.get('star_count', 0)} stars)\n"
                f"   {project.get('description', 'No description')}\n"
                f"   URL: {project.get('web_url', '')}"
            )
        return ToolResult(output="\n\n".join(results))

    async def _get_project(self, owner: str = "", repo: str = "", **kwargs: Any) -> ToolResult:
        if not owner or not repo:
            return ToolResult(output="", error="Both 'owner' and 'repo' are required.")

        from urllib.parse import quote_plus

        project_id = quote_plus(f"{owner}/{repo}")
        data = await self._api_get(f"{self._url}/api/v4/projects/{project_id}")
        return ToolResult(
            output=(
                f"**{data['path_with_namespace']}** ({data.get('star_count', 0)} stars, "
                f"{data.get('forks_count', 0)} forks)\n"
                f"Description: {data.get('description', 'None')}\n"
                f"URL: {data.get('web_url', '')}"
            )
        )

    async def _search_issues(self, query: str = "", **kwargs: Any) -> ToolResult:
        if not query:
            return ToolResult(output="", error="No query provided for search_issues.")

        from urllib.parse import quote_plus

        data = await self._api_get(
            f"{self._url}/api/v4/issues?search={quote_plus(query)}&per_page=5"
        )
        if not data:
            return ToolResult(output="No issues found.")

        results = []
        for i, issue in enumerate(data, 1):
            results.append(
                f"{i}. **{issue['title']}** (#{issue['iid']})\n"
                f"   State: {issue.get('state', 'unknown')}\n"
                f"   URL: {issue.get('web_url', '')}"
            )
        return ToolResult(output="\n\n".join(results))

    async def _get_issue(
        self, owner: str = "", repo: str = "", issue_number: int = 0, **kwargs: Any
    ) -> ToolResult:
        if not owner or not repo or not issue_number:
            return ToolResult(output="", error="'owner', 'repo', and 'issue_number' are required.")

        from urllib.parse import quote_plus

        project_id = quote_plus(f"{owner}/{repo}")
        data = await self._api_get(
            f"{self._url}/api/v4/projects/{project_id}/issues/{issue_number}"
        )
        body = (data.get("description") or "No description")[:500]
        return ToolResult(
            output=(
                f"**{data['title']}** (#{data['iid']})\n"
                f"State: {data.get('state', 'unknown')}\n"
                f"Author: {data.get('author', {}).get('username', 'unknown')}\n"
                f"URL: {data.get('web_url', '')}\n\n"
                f"{body}"
            )
        )
