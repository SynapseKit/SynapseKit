"""Tests for v0.6.9 features: Slack, Jira, Brave Search tools + approval_node, dynamic_route_node."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ------------------------------------------------------------------ #
# Version
# ------------------------------------------------------------------ #


def test_version():
    import synapsekit

    assert synapsekit.__version__ == "0.9.0"


# ================================================================== #
# Feature 1: SlackTool
# ================================================================== #


class TestSlackTool:
    def test_import(self):
        from synapsekit import SlackTool

        assert SlackTool is not None

    async def test_no_action_error(self):
        from synapsekit import SlackTool

        tool = SlackTool(webhook_url="https://hooks.slack.com/x")
        result = await tool.run()
        assert result.is_error
        assert "No action" in result.error

    async def test_no_text_error(self):
        from synapsekit import SlackTool

        tool = SlackTool(webhook_url="https://hooks.slack.com/x")
        result = await tool.run(action="send_webhook")
        assert result.is_error
        assert "No text" in result.error

    async def test_unknown_action(self):
        from synapsekit import SlackTool

        tool = SlackTool(webhook_url="https://hooks.slack.com/x")
        result = await tool.run(action="delete_channel", text="hi")
        assert result.is_error
        assert "Unknown action" in result.error

    async def test_send_webhook_no_url(self):
        from synapsekit import SlackTool

        tool = SlackTool()
        result = await tool.run(action="send_webhook", text="hello")
        assert result.is_error
        assert "SLACK_WEBHOOK_URL" in result.error

    async def test_send_message_no_token(self):
        from synapsekit import SlackTool

        tool = SlackTool()
        result = await tool.run(action="send_message", text="hi", channel="#general")
        assert result.is_error
        assert "SLACK_BOT_TOKEN" in result.error

    async def test_send_message_no_channel(self):
        from synapsekit import SlackTool

        tool = SlackTool(bot_token="xoxb-test")
        result = await tool.run(action="send_message", text="hi")
        assert result.is_error
        assert "No channel" in result.error

    async def test_send_webhook_success(self):
        from synapsekit import SlackTool

        tool = SlackTool(webhook_url="https://hooks.slack.com/services/T/B/X")
        with patch.object(tool, "_post_json", new_callable=AsyncMock, return_value="ok"):
            result = await tool.run(action="send_webhook", text="Hello!")
        assert not result.is_error
        assert "webhook" in result.output.lower()

    async def test_send_message_success(self):
        from synapsekit import SlackTool

        tool = SlackTool(bot_token="xoxb-test-token")
        resp = json.dumps({"ok": True, "channel": "C123", "ts": "1234567890.123456"})
        with patch.object(tool, "_post_json", new_callable=AsyncMock, return_value=resp):
            result = await tool.run(action="send_message", channel="#general", text="Hi!")
        assert not result.is_error
        assert "#general" in result.output

    async def test_send_message_api_error(self):
        from synapsekit import SlackTool

        tool = SlackTool(bot_token="xoxb-test-token")
        resp = json.dumps({"ok": False, "error": "channel_not_found"})
        with patch.object(tool, "_post_json", new_callable=AsyncMock, return_value=resp):
            result = await tool.run(action="send_message", channel="#nope", text="Hi!")
        assert result.is_error
        assert "channel_not_found" in result.error

    def test_name_and_description(self):
        from synapsekit import SlackTool

        tool = SlackTool()
        assert tool.name == "slack"
        assert "Slack" in tool.description


# ================================================================== #
# Feature 2: JiraTool
# ================================================================== #


class TestJiraTool:
    def test_import(self):
        from synapsekit import JiraTool

        assert JiraTool is not None

    async def test_no_action_error(self):
        from synapsekit import JiraTool

        tool = JiraTool(url="https://x.atlassian.net", email="a@b.com", api_token="tok")
        result = await tool.run()
        assert result.is_error
        assert "No action" in result.error

    async def test_unknown_action(self):
        from synapsekit import JiraTool

        tool = JiraTool(url="https://x.atlassian.net", email="a@b.com", api_token="tok")
        result = await tool.run(action="delete_project")
        assert result.is_error
        assert "Unknown action" in result.error

    async def test_no_url_error(self):
        from synapsekit import JiraTool

        tool = JiraTool()
        result = await tool.run(action="search_issues", query="project=TEST")
        assert result.is_error
        assert "JIRA_URL" in result.error

    async def test_no_credentials_error(self):
        from synapsekit import JiraTool

        tool = JiraTool(url="https://x.atlassian.net")
        result = await tool.run(action="search_issues", query="project=TEST")
        assert result.is_error
        assert "JIRA_EMAIL" in result.error

    async def test_search_no_query(self):
        from synapsekit import JiraTool

        tool = JiraTool(url="https://x.atlassian.net", email="a@b.com", api_token="tok")
        result = await tool.run(action="search_issues")
        assert result.is_error
        assert "No JQL" in result.error

    async def test_get_issue_no_key(self):
        from synapsekit import JiraTool

        tool = JiraTool(url="https://x.atlassian.net", email="a@b.com", api_token="tok")
        result = await tool.run(action="get_issue")
        assert result.is_error
        assert "issue_key" in result.error

    async def test_create_issue_missing_fields(self):
        from synapsekit import JiraTool

        tool = JiraTool(url="https://x.atlassian.net", email="a@b.com", api_token="tok")
        result = await tool.run(action="create_issue")
        assert result.is_error
        assert "project_key" in result.error

    async def test_add_comment_missing_fields(self):
        from synapsekit import JiraTool

        tool = JiraTool(url="https://x.atlassian.net", email="a@b.com", api_token="tok")
        result = await tool.run(action="add_comment")
        assert result.is_error
        assert "issue_key" in result.error

    async def test_search_issues_success(self):
        from synapsekit import JiraTool

        tool = JiraTool(url="https://x.atlassian.net", email="a@b.com", api_token="tok")
        mock_data = {
            "issues": [
                {
                    "key": "PROJ-1",
                    "fields": {
                        "summary": "Fix login bug",
                        "status": {"name": "Open"},
                    },
                }
            ]
        }
        with patch.object(tool, "_api", new_callable=AsyncMock, return_value=mock_data):
            result = await tool.run(action="search_issues", query="project=PROJ")
        assert not result.is_error
        assert "PROJ-1" in result.output
        assert "Fix login bug" in result.output

    async def test_search_issues_empty(self):
        from synapsekit import JiraTool

        tool = JiraTool(url="https://x.atlassian.net", email="a@b.com", api_token="tok")
        with patch.object(tool, "_api", new_callable=AsyncMock, return_value={"issues": []}):
            result = await tool.run(action="search_issues", query="project=NONE")
        assert result.output == "No issues found."

    async def test_get_issue_success(self):
        from synapsekit import JiraTool

        tool = JiraTool(url="https://x.atlassian.net", email="a@b.com", api_token="tok")
        mock_data = {
            "key": "PROJ-42",
            "fields": {
                "summary": "Add dark mode",
                "status": {"name": "In Progress"},
                "issuetype": {"name": "Story"},
                "assignee": {"displayName": "Alice"},
                "description": "We need dark mode for the app.",
            },
        }
        with patch.object(tool, "_api", new_callable=AsyncMock, return_value=mock_data):
            result = await tool.run(action="get_issue", issue_key="PROJ-42")
        assert "PROJ-42" in result.output
        assert "Alice" in result.output

    async def test_create_issue_success(self):
        from synapsekit import JiraTool

        tool = JiraTool(url="https://x.atlassian.net", email="a@b.com", api_token="tok")
        with patch.object(tool, "_api", new_callable=AsyncMock, return_value={"key": "PROJ-99"}):
            result = await tool.run(
                action="create_issue", project_key="PROJ", summary="New feature"
            )
        assert "PROJ-99" in result.output

    async def test_add_comment_success(self):
        from synapsekit import JiraTool

        tool = JiraTool(url="https://x.atlassian.net", email="a@b.com", api_token="tok")
        with patch.object(tool, "_api", new_callable=AsyncMock, return_value={}):
            result = await tool.run(action="add_comment", issue_key="PROJ-1", comment="Looks good!")
        assert "PROJ-1" in result.output

    def test_name_and_description(self):
        from synapsekit import JiraTool

        tool = JiraTool()
        assert tool.name == "jira"
        assert "Jira" in tool.description


# ================================================================== #
# Feature 3: BraveSearchTool
# ================================================================== #


class TestBraveSearchTool:
    def test_import(self):
        from synapsekit import BraveSearchTool

        assert BraveSearchTool is not None

    async def test_no_query_error(self):
        from synapsekit import BraveSearchTool

        tool = BraveSearchTool(api_key="BSA-test")
        result = await tool.run()
        assert result.is_error
        assert "No search query" in result.error

    async def test_no_api_key_error(self):
        from synapsekit import BraveSearchTool

        tool = BraveSearchTool()
        result = await tool.run(query="test")
        assert result.is_error
        assert "BRAVE_API_KEY" in result.error

    async def test_search_success(self):
        from synapsekit import BraveSearchTool

        tool = BraveSearchTool(api_key="BSA-test")
        mock_resp = {
            "web": {
                "results": [
                    {
                        "title": "AI News",
                        "url": "https://example.com/ai",
                        "description": "Latest AI developments",
                    },
                    {
                        "title": "ML Guide",
                        "url": "https://example.com/ml",
                        "description": "Machine learning tutorial",
                    },
                ]
            }
        }

        async def mock_run(query="", count=5, **kw):
            from synapsekit.agents.base import ToolResult

            results = []
            for i, r in enumerate(mock_resp["web"]["results"], 1):
                results.append(f"{i}. **{r['title']}**\n   URL: {r['url']}\n   {r['description']}")
            return ToolResult(output="\n\n".join(results))

        with patch.object(tool, "run", side_effect=mock_run):
            result = await tool.run(query="AI news")
        assert "AI News" in result.output
        assert "ML Guide" in result.output

    async def test_search_no_results(self):
        from synapsekit import BraveSearchTool

        tool = BraveSearchTool(api_key="BSA-test")
        mock_resp = {"web": {"results": []}}

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(mock_resp).encode()
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            result = await tool.run(query="xyznonexistent12345")
        assert result.output == "No results found."

    async def test_input_kwarg_fallback(self):
        from synapsekit import BraveSearchTool

        tool = BraveSearchTool(api_key="BSA-test")
        mock_resp = {"web": {"results": []}}

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(mock_resp).encode()
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            result = await tool.run(input="test query")
        assert result.output == "No results found."

    def test_name_and_description(self):
        from synapsekit import BraveSearchTool

        tool = BraveSearchTool()
        assert tool.name == "brave_search"
        assert "Brave" in tool.description


# ================================================================== #
# Feature 4: approval_node
# ================================================================== #


class TestApprovalNode:
    def test_import(self):
        from synapsekit import approval_node

        assert callable(approval_node)

    async def test_approved_passes_through(self):
        from synapsekit import approval_node

        node = approval_node(approval_key="approved")
        state = {"approved": True, "data": "hello"}
        result = await node(state)
        assert result == state

    async def test_not_approved_raises_interrupt(self):
        from synapsekit import GraphInterrupt, approval_node

        node = approval_node(approval_key="approved", message="Please approve this.")
        state = {"approved": False, "data": "hello"}
        with pytest.raises(GraphInterrupt) as exc_info:
            await node(state)
        assert "Please approve this." in str(exc_info.value)

    async def test_missing_key_raises_interrupt(self):
        from synapsekit import GraphInterrupt, approval_node

        node = approval_node(approval_key="human_ok")
        state = {"data": "hello"}
        with pytest.raises(GraphInterrupt):
            await node(state)

    async def test_dynamic_message(self):
        from synapsekit import GraphInterrupt, approval_node

        node = approval_node(
            approval_key="ok",
            message=lambda s: f"Review: {s.get('draft', 'N/A')}",
        )
        state = {"ok": False, "draft": "Hello world"}
        with pytest.raises(GraphInterrupt) as exc_info:
            await node(state)
        assert "Review: Hello world" in str(exc_info.value)

    async def test_interrupt_includes_state_data(self):
        from synapsekit import GraphInterrupt, approval_node

        node = approval_node(approval_key="ok", data={"extra": "info"})
        state = {"ok": False, "foo": "bar"}
        with pytest.raises(GraphInterrupt) as exc_info:
            await node(state)
        assert exc_info.value.data["extra"] == "info"
        assert exc_info.value.data["foo"] == "bar"

    async def test_truthy_values_pass(self):
        from synapsekit import approval_node

        node = approval_node(approval_key="approved")
        for truthy in [True, 1, "yes", [1], {"a": 1}]:
            result = await node({"approved": truthy})
            assert result["approved"] == truthy

    async def test_falsy_values_interrupt(self):
        from synapsekit import GraphInterrupt, approval_node

        node = approval_node(approval_key="approved")
        for falsy in [False, 0, "", None, [], {}]:
            with pytest.raises(GraphInterrupt):
                await node({"approved": falsy})

    async def test_default_message(self):
        from synapsekit import GraphInterrupt, approval_node

        node = approval_node()
        with pytest.raises(GraphInterrupt) as exc_info:
            await node({})
        assert "Approval required" in str(exc_info.value)


# ================================================================== #
# Feature 5: dynamic_route_node
# ================================================================== #


class TestDynamicRouteNode:
    def test_import(self):
        from synapsekit import dynamic_route_node

        assert callable(dynamic_route_node)

    async def test_routes_to_correct_subgraph(self):
        from synapsekit import dynamic_route_node

        fast_graph = AsyncMock()
        fast_graph.run.return_value = {"output": "fast result"}
        slow_graph = AsyncMock()
        slow_graph.run.return_value = {"output": "slow result"}

        node = dynamic_route_node(
            routing_fn=lambda s: "fast" if s.get("urgent") else "slow",
            subgraphs={"fast": fast_graph, "slow": slow_graph},
        )

        result = await node({"urgent": True, "input": "query"})
        assert result["output"] == "fast result"
        fast_graph.run.assert_called_once()

    async def test_routes_to_slow_path(self):
        from synapsekit import dynamic_route_node

        fast_graph = AsyncMock()
        slow_graph = AsyncMock()
        slow_graph.run.return_value = {"output": "slow result"}

        node = dynamic_route_node(
            routing_fn=lambda s: "fast" if s.get("urgent") else "slow",
            subgraphs={"fast": fast_graph, "slow": slow_graph},
        )

        result = await node({"urgent": False, "input": "query"})
        assert result["output"] == "slow result"
        slow_graph.run.assert_called_once()
        fast_graph.run.assert_not_called()

    async def test_unknown_route_raises_error(self):
        from synapsekit import dynamic_route_node

        node = dynamic_route_node(
            routing_fn=lambda s: "unknown",
            subgraphs={"a": AsyncMock(), "b": AsyncMock()},
        )
        with pytest.raises(ValueError, match="Unknown route key"):
            await node({"input": "test"})

    async def test_input_mapping(self):
        from synapsekit import dynamic_route_node

        sub = AsyncMock()
        sub.run.return_value = {"result": "done"}

        node = dynamic_route_node(
            routing_fn=lambda s: "only",
            subgraphs={"only": sub},
            input_mapping={"parent_input": "input"},
        )

        await node({"parent_input": "hello", "extra": "ignored"})
        sub.run.assert_called_once_with({"input": "hello"})

    async def test_output_mapping(self):
        from synapsekit import dynamic_route_node

        sub = AsyncMock()
        sub.run.return_value = {"output": "result"}

        node = dynamic_route_node(
            routing_fn=lambda s: "only",
            subgraphs={"only": sub},
            output_mapping={"output": "parent_result"},
        )

        result = await node({"input": "test"})
        assert result == {"parent_result": "result"}

    async def test_async_routing_fn(self):
        from synapsekit import dynamic_route_node

        sub_a = AsyncMock()
        sub_a.run.return_value = {"output": "a"}

        async def async_router(state):
            return "a"

        node = dynamic_route_node(
            routing_fn=async_router,
            subgraphs={"a": sub_a, "b": AsyncMock()},
        )

        result = await node({"input": "test"})
        assert result["output"] == "a"

    async def test_no_mappings_passes_full_state(self):
        from synapsekit import dynamic_route_node

        sub = AsyncMock()
        sub.run.return_value = {"x": 1, "y": 2}

        node = dynamic_route_node(
            routing_fn=lambda s: "only",
            subgraphs={"only": sub},
        )

        state = {"a": 10, "b": 20}
        result = await node(state)
        sub.run.assert_called_once_with({"a": 10, "b": 20})
        assert result == {"x": 1, "y": 2}

    async def test_multiple_routes(self):
        from synapsekit import dynamic_route_node

        graphs = {}
        for name in ["email", "slack", "sms"]:
            g = AsyncMock()
            g.run.return_value = {"sent_via": name}
            graphs[name] = g

        node = dynamic_route_node(
            routing_fn=lambda s: s.get("channel", "email"),
            subgraphs=graphs,
        )

        for channel in ["email", "slack", "sms"]:
            result = await node({"channel": channel, "msg": "hi"})
            assert result["sent_via"] == channel
