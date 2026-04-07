"""Linear Tool: manage issues via the Linear GraphQL API."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.request
from collections.abc import Callable, Coroutine
from typing import Any

from ..base import BaseTool, ToolResult

log = logging.getLogger(__name__)

_LINEAR_API = "https://api.linear.app/graphql"


class LinearTool(BaseTool):
    """Manage Linear issues via GraphQL API.

    Auth via constructor arg or ``LINEAR_API_KEY`` env var.
    Uses stdlib ``urllib`` only — no extra dependencies.

    Usage::

        tool = LinearTool(api_key="lin_api_...")
        result = await tool.run(action="list_issues", team_id="TEAM-ID")
    """

    name = "linear"
    description = (
        "Manage Linear issues. Actions: list_issues, get_issue, create_issue, update_issue."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list_issues", "get_issue", "create_issue", "update_issue"],
                "description": "Action to perform",
            },
            "team_id": {"type": "string", "description": "Linear team ID"},
            "issue_id": {"type": "string", "description": "Linear issue ID"},
            "title": {"type": "string", "description": "Issue title"},
            "description": {"type": "string", "description": "Issue description"},
            "priority": {
                "type": "integer",
                "description": "Priority (0=none, 1=urgent, 2=high, 3=medium, 4=low)",
            },
            "status": {"type": "string", "description": "New status name"},
            "filter": {"type": "string", "description": "Filter string for issues"},
        },
        "required": ["action"],
    }

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("LINEAR_API_KEY", "")

    async def run(self, action: str = "", **kwargs: Any) -> ToolResult:
        if not action:
            return ToolResult(output="", error="No action specified.")

        _handler_t = Callable[..., Coroutine[Any, Any, ToolResult]]
        handlers: dict[str, _handler_t] = {
            "list_issues": self._list_issues,
            "get_issue": self._get_issue,
            "create_issue": self._create_issue,
            "update_issue": self._update_issue,
        }

        handler = handlers.get(action)
        if handler is None:
            return ToolResult(
                output="",
                error=f"Unknown action: {action}. Must be one of: {', '.join(handlers)}",
            )

        if not self._api_key:
            return ToolResult(output="", error="LINEAR_API_KEY is required.")

        try:
            return await handler(**kwargs)
        except Exception as e:
            return ToolResult(output="", error=f"Linear API error: {e}")

    async def _graphql(self, query: str, variables: dict[str, Any] | None = None) -> Any:
        payload = json.dumps({"query": query, "variables": variables or {}}).encode()
        req = urllib.request.Request(
            _LINEAR_API,
            data=payload,
            headers={
                "Authorization": self._api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )

        loop = asyncio.get_running_loop()

        def _fetch():
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = json.loads(resp.read().decode())
            if "errors" in body:
                raise RuntimeError(body["errors"][0].get("message", "GraphQL error"))
            return body.get("data", {})

        return await loop.run_in_executor(None, _fetch)

    async def _list_issues(self, team_id: str = "", filter: str = "", **kw: Any) -> ToolResult:
        if not team_id:
            return ToolResult(output="", error="team_id is required for list_issues.")

        q = """
        query($teamId: String!) {
            team(id: $teamId) {
                issues(first: 20) {
                    nodes {
                        id
                        title
                        state { name }
                        assignee { name }
                        priority
                    }
                }
            }
        }
        """
        data = await self._graphql(q, {"teamId": team_id})
        nodes = data.get("team", {}).get("issues", {}).get("nodes", [])
        if not nodes:
            return ToolResult(output="No issues found.")

        lines = []
        for i, n in enumerate(nodes, 1):
            assignee = (n.get("assignee") or {}).get("name", "Unassigned")
            state = (n.get("state") or {}).get("name", "Unknown")
            lines.append(f"{i}. **{n['title']}** (Status: {state}, Assignee: {assignee})")
        return ToolResult(output="\n".join(lines))

    async def _get_issue(self, issue_id: str = "", **kw: Any) -> ToolResult:
        if not issue_id:
            return ToolResult(output="", error="issue_id is required for get_issue.")

        q = """
        query($issueId: String!) {
            issue(id: $issueId) {
                id
                title
                description
                state { name }
                assignee { name }
                priority
                createdAt
            }
        }
        """
        data = await self._graphql(q, {"issueId": issue_id})
        issue = data.get("issue", {})
        if not issue:
            return ToolResult(output="", error="Issue not found.")

        desc = (issue.get("description") or "No description")[:500]
        assignee = (issue.get("assignee") or {}).get("name", "Unassigned")
        state = (issue.get("state") or {}).get("name", "Unknown")
        return ToolResult(
            output=(
                f"**{issue.get('title', 'Untitled')}**\n"
                f"Status: {state}\n"
                f"Assignee: {assignee}\n"
                f"Priority: {issue.get('priority', 0)}\n"
                f"Created: {issue.get('createdAt', 'N/A')}\n\n"
                f"{desc}"
            )
        )

    async def _create_issue(
        self,
        team_id: str = "",
        title: str = "",
        description: str = "",
        priority: int = 0,
        **kw: Any,
    ) -> ToolResult:
        if not team_id or not title:
            return ToolResult(output="", error="team_id and title are required for create_issue.")

        q = """
        mutation($teamId: String!, $title: String!, $description: String, $priority: Int) {
            issueCreate(input: {
                teamId: $teamId
                title: $title
                description: $description
                priority: $priority
            }) {
                issue { id title }
            }
        }
        """
        data = await self._graphql(
            q, {"teamId": team_id, "title": title, "description": description, "priority": priority}
        )
        issue = data.get("issueCreate", {}).get("issue", {})
        return ToolResult(
            output=f"Created issue **{issue.get('title', title)}** ({issue.get('id', 'unknown')})."
        )

    async def _update_issue(self, issue_id: str = "", status: str = "", **kw: Any) -> ToolResult:
        if not issue_id or not status:
            return ToolResult(output="", error="issue_id and status are required for update_issue.")

        q = """
        mutation($issueId: String!, $stateId: String!) {
            issueUpdate(id: $issueId, input: { stateId: $stateId }) {
                issue {
                    id
                    title
                    state { name }
                }
            }
        }
        """
        data = await self._graphql(q, {"issueId": issue_id, "stateId": status})
        issue = data.get("issueUpdate", {}).get("issue", {})
        new_state = (issue.get("state") or {}).get("name", status)
        return ToolResult(
            output=f"Updated **{issue.get('title', issue_id)}** to status: {new_state}."
        )
