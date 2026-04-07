from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from synapsekit.agents.tools.linear import LinearTool


def _make_tool(**kw):
    defaults = {"api_key": "lin_api_test123"}
    defaults.update(kw)
    return LinearTool(**defaults)


def _mock_urlopen(data):
    resp = MagicMock()
    resp.read.return_value = json.dumps({"data": data}).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestLinearTool:
    @pytest.mark.asyncio
    async def test_no_action(self):
        tool = _make_tool()
        res = await tool.run()
        assert res.is_error
        assert "No action" in res.error

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        tool = _make_tool()
        res = await tool.run(action="delete_issue")
        assert res.is_error
        assert "Unknown action" in res.error

    @pytest.mark.asyncio
    async def test_missing_api_key(self):
        tool = LinearTool(api_key="")
        res = await tool.run(action="list_issues", team_id="T1")
        assert res.is_error
        assert "LINEAR_API_KEY" in res.error

    @pytest.mark.asyncio
    async def test_list_issues(self):
        data = {
            "team": {
                "issues": {
                    "nodes": [
                        {
                            "id": "ISS-1",
                            "title": "Fix login bug",
                            "state": {"name": "In Progress"},
                            "assignee": {"name": "Alice"},
                            "priority": 2,
                        }
                    ]
                }
            }
        }
        tool = _make_tool()
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(data)):
            res = await tool.run(action="list_issues", team_id="team-abc")
        assert not res.is_error
        assert "Fix login bug" in res.output
        assert "In Progress" in res.output

    @pytest.mark.asyncio
    async def test_list_issues_no_team_id(self):
        tool = _make_tool()
        res = await tool.run(action="list_issues")
        assert res.is_error
        assert "team_id" in res.error

    @pytest.mark.asyncio
    async def test_list_issues_empty(self):
        data = {"team": {"issues": {"nodes": []}}}
        tool = _make_tool()
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(data)):
            res = await tool.run(action="list_issues", team_id="team-abc")
        assert "No issues found" in res.output

    @pytest.mark.asyncio
    async def test_get_issue(self):
        data = {
            "issue": {
                "id": "ISS-42",
                "title": "Add dark mode",
                "description": "We need dark mode support",
                "state": {"name": "Todo"},
                "assignee": {"name": "Bob"},
                "priority": 3,
                "createdAt": "2026-01-15T10:00:00Z",
            }
        }
        tool = _make_tool()
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(data)):
            res = await tool.run(action="get_issue", issue_id="ISS-42")
        assert not res.is_error
        assert "Add dark mode" in res.output
        assert "Bob" in res.output

    @pytest.mark.asyncio
    async def test_get_issue_no_id(self):
        tool = _make_tool()
        res = await tool.run(action="get_issue")
        assert res.is_error
        assert "issue_id" in res.error

    @pytest.mark.asyncio
    async def test_create_issue(self):
        data = {"issueCreate": {"issue": {"id": "ISS-99", "title": "New feature"}}}
        tool = _make_tool()
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(data)):
            res = await tool.run(action="create_issue", team_id="team-abc", title="New feature")
        assert not res.is_error
        assert "New feature" in res.output
        assert "ISS-99" in res.output

    @pytest.mark.asyncio
    async def test_create_issue_missing_params(self):
        tool = _make_tool()
        res = await tool.run(action="create_issue", team_id="team-abc")
        assert res.is_error
        assert "title" in res.error

    @pytest.mark.asyncio
    async def test_update_issue(self):
        data = {
            "issueUpdate": {
                "issue": {"id": "ISS-42", "title": "Fix bug", "state": {"name": "Done"}}
            }
        }
        tool = _make_tool()
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(data)):
            res = await tool.run(action="update_issue", issue_id="ISS-42", status="done-state-id")
        assert not res.is_error
        assert "Done" in res.output

    @pytest.mark.asyncio
    async def test_update_issue_missing_params(self):
        tool = _make_tool()
        res = await tool.run(action="update_issue", issue_id="ISS-42")
        assert res.is_error
        assert "status" in res.error

    @pytest.mark.asyncio
    async def test_api_error_handled(self):
        tool = _make_tool()
        with patch("urllib.request.urlopen", side_effect=Exception("connection refused")):
            res = await tool.run(action="list_issues", team_id="team-abc")
        assert res.is_error
        assert "Linear API error" in res.error

    @pytest.mark.asyncio
    async def test_graphql_error_propagated(self):
        resp = MagicMock()
        resp.read.return_value = json.dumps({"errors": [{"message": "Not authorized"}]}).encode()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)

        tool = _make_tool()
        with patch("urllib.request.urlopen", return_value=resp):
            res = await tool.run(action="list_issues", team_id="team-abc")
        assert res.is_error
        assert "Not authorized" in res.error

    def test_schema(self):
        tool = _make_tool()
        s = tool.schema()
        assert s["function"]["name"] == "linear"
        props = s["function"]["parameters"]["properties"]
        assert "action" in props
        assert "team_id" in props
        assert "issue_id" in props

    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("LINEAR_API_KEY", "lin_env_key")
        tool = LinearTool()
        assert tool._api_key == "lin_env_key"
