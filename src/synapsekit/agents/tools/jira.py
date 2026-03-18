"""Jira Tool: interact with the Jira REST API v2."""

from __future__ import annotations

import os
from typing import Any

from ..base import BaseTool, ToolResult


class JiraTool(BaseTool):
    """Interact with Jira REST API v2.

    Supports searching issues (JQL), getting issue details, creating issues,
    and adding comments.  Uses stdlib ``urllib`` + ``base64`` Basic auth — no
    extra dependencies.

    Usage::

        tool = JiraTool(url="https://myco.atlassian.net", email="me@co.com", api_token="...")
        result = await tool.run(action="search_issues", query="project=PROJ AND status=Open")
    """

    name = "jira"
    description = (
        "Interact with Jira to search issues, get issue details, create issues, and add comments."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "Action to perform",
                "enum": ["search_issues", "get_issue", "create_issue", "add_comment"],
            },
            "query": {
                "type": "string",
                "description": "JQL query (for search_issues)",
            },
            "issue_key": {
                "type": "string",
                "description": "Issue key e.g. PROJ-123 (for get_issue, add_comment)",
            },
            "project_key": {
                "type": "string",
                "description": "Project key (for create_issue)",
            },
            "summary": {
                "type": "string",
                "description": "Issue summary (for create_issue)",
            },
            "description": {
                "type": "string",
                "description": "Issue description (for create_issue)",
            },
            "issue_type": {
                "type": "string",
                "description": "Issue type e.g. Task, Bug (for create_issue, default: Task)",
                "default": "Task",
            },
            "comment": {
                "type": "string",
                "description": "Comment body (for add_comment)",
            },
        },
        "required": ["action"],
    }

    def __init__(
        self,
        url: str | None = None,
        email: str | None = None,
        api_token: str | None = None,
    ) -> None:
        self._url = (url or os.environ.get("JIRA_URL", "")).rstrip("/")
        self._email = email or os.environ.get("JIRA_EMAIL")
        self._api_token = api_token or os.environ.get("JIRA_API_TOKEN")

    async def run(self, action: str = "", **kwargs: Any) -> ToolResult:
        if not action:
            return ToolResult(output="", error="No action specified.")

        handlers = {
            "search_issues": self._search_issues,
            "get_issue": self._get_issue,
            "create_issue": self._create_issue,
            "add_comment": self._add_comment,
        }

        handler = handlers.get(action)
        if handler is None:
            return ToolResult(
                output="",
                error=f"Unknown action: {action}. Must be one of: {', '.join(handlers)}",
            )

        if not self._url:
            return ToolResult(output="", error="No JIRA_URL configured.")
        if not self._email or not self._api_token:
            return ToolResult(output="", error="JIRA_EMAIL and JIRA_API_TOKEN are required.")

        try:
            return await handler(**kwargs)
        except Exception as e:
            return ToolResult(output="", error=f"Jira API error: {e}")

    def _auth_header(self) -> str:
        import base64

        creds = base64.b64encode(f"{self._email}:{self._api_token}".encode()).decode()
        return f"Basic {creds}"

    def _build_request(self, url: str, *, method: str = "GET", data: bytes | None = None) -> Any:
        import urllib.request

        headers = {
            "Authorization": self._auth_header(),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        return urllib.request.Request(url, data=data, headers=headers, method=method)

    async def _api(
        self, path: str, *, method: str = "GET", payload: dict[str, Any] | None = None
    ) -> Any:
        import asyncio
        import json
        import urllib.request

        url = f"{self._url}/rest/api/2{path}"
        data = json.dumps(payload).encode() if payload else None
        req = self._build_request(url, method=method, data=data)

        loop = asyncio.get_event_loop()

        def _fetch():
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = resp.read().decode()
                return json.loads(body) if body else {}

        return await loop.run_in_executor(None, _fetch)

    async def _search_issues(self, query: str = "", **kwargs: Any) -> ToolResult:
        if not query:
            return ToolResult(output="", error="No JQL query provided for search_issues.")

        from urllib.parse import quote_plus

        data = await self._api(f"/search?jql={quote_plus(query)}&maxResults=10")
        issues = data.get("issues", [])
        if not issues:
            return ToolResult(output="No issues found.")

        results = []
        for i, issue in enumerate(issues, 1):
            key = issue["key"]
            fields = issue.get("fields", {})
            summary = fields.get("summary", "No summary")
            status = fields.get("status", {}).get("name", "Unknown")
            results.append(f"{i}. **{key}** — {summary} (Status: {status})")
        return ToolResult(output="\n".join(results))

    async def _get_issue(self, issue_key: str = "", **kwargs: Any) -> ToolResult:
        if not issue_key:
            return ToolResult(output="", error="No issue_key provided for get_issue.")

        data = await self._api(f"/issue/{issue_key}")
        fields = data.get("fields", {})
        desc = (fields.get("description") or "No description")[:500]
        return ToolResult(
            output=(
                f"**{data['key']}** — {fields.get('summary', 'No summary')}\n"
                f"Status: {fields.get('status', {}).get('name', 'Unknown')}\n"
                f"Type: {fields.get('issuetype', {}).get('name', 'Unknown')}\n"
                f"Assignee: {(fields.get('assignee') or {}).get('displayName', 'Unassigned')}\n\n"
                f"{desc}"
            )
        )

    async def _create_issue(
        self,
        project_key: str = "",
        summary: str = "",
        description: str = "",
        issue_type: str = "Task",
        **kwargs: Any,
    ) -> ToolResult:
        if not project_key or not summary:
            return ToolResult(
                output="", error="project_key and summary are required for create_issue."
            )

        payload = {
            "fields": {
                "project": {"key": project_key},
                "summary": summary,
                "description": description,
                "issuetype": {"name": issue_type},
            }
        }
        data = await self._api("/issue", method="POST", payload=payload)
        return ToolResult(output=f"Created issue **{data['key']}**.")

    async def _add_comment(
        self, issue_key: str = "", comment: str = "", **kwargs: Any
    ) -> ToolResult:
        if not issue_key or not comment:
            return ToolResult(
                output="", error="issue_key and comment are required for add_comment."
            )

        await self._api(
            f"/issue/{issue_key}/comment",
            method="POST",
            payload={"body": comment},
        )
        return ToolResult(output=f"Comment added to **{issue_key}**.")
