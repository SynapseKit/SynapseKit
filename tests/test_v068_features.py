"""Tests for v0.6.8 features: 5 tools, execution trace, WebSocket streaming."""

from __future__ import annotations

import json
import sys
import time
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ------------------------------------------------------------------ #
# Version
# ------------------------------------------------------------------ #


def test_version():
    import synapsekit

    assert synapsekit.__version__ == "0.7.0"


# ------------------------------------------------------------------ #
# Feature 1: VectorSearchTool
# ------------------------------------------------------------------ #


class TestVectorSearchTool:
    def test_import(self):
        from synapsekit import VectorSearchTool

        assert VectorSearchTool is not None

    async def test_query_returns_results(self):
        from synapsekit import VectorSearchTool

        retriever = AsyncMock()
        retriever.retrieve.return_value = ["Document about AI", "Document about ML"]

        tool = VectorSearchTool(retriever)
        result = await tool.run(query="machine learning")

        retriever.retrieve.assert_called_once_with("machine learning", top_k=5)
        assert "Document about AI" in result.output
        assert "Document about ML" in result.output
        assert not result.is_error

    async def test_empty_results(self):
        from synapsekit import VectorSearchTool

        retriever = AsyncMock()
        retriever.retrieve.return_value = []

        tool = VectorSearchTool(retriever)
        result = await tool.run(query="something obscure")

        assert result.output == "No results found."

    async def test_no_query_error(self):
        from synapsekit import VectorSearchTool

        retriever = AsyncMock()
        tool = VectorSearchTool(retriever)
        result = await tool.run()

        assert result.is_error
        assert "No search query" in result.error

    async def test_custom_name_and_description(self):
        from synapsekit import VectorSearchTool

        retriever = AsyncMock()
        tool = VectorSearchTool(retriever, name="my_kb", description="My knowledge base")

        assert tool.name == "my_kb"
        assert tool.description == "My knowledge base"

    async def test_custom_top_k(self):
        from synapsekit import VectorSearchTool

        retriever = AsyncMock()
        retriever.retrieve.return_value = ["doc1"]
        tool = VectorSearchTool(retriever)
        await tool.run(query="test", top_k=10)

        retriever.retrieve.assert_called_once_with("test", top_k=10)

    def test_schema(self):
        from synapsekit import VectorSearchTool

        tool = VectorSearchTool(AsyncMock())
        schema = tool.schema()
        assert schema["function"]["name"] == "vector_search"
        assert "query" in schema["function"]["parameters"]["properties"]


# ------------------------------------------------------------------ #
# Feature 2: PubMedSearchTool
# ------------------------------------------------------------------ #


def _make_esearch_xml(pmids: list[str]) -> bytes:
    ids = "".join(f"<Id>{p}</Id>" for p in pmids)
    return f"<eSearchResult><IdList>{ids}</IdList></eSearchResult>".encode()


def _make_efetch_xml(articles: list[dict]) -> bytes:
    parts = []
    for a in articles:
        authors = "".join(
            f"<Author><LastName>{au.split()[0]}</LastName>"
            f"<ForeName>{' '.join(au.split()[1:])}</ForeName></Author>"
            for au in a.get("authors", [])
        )
        parts.append(
            f"<PubmedArticle><MedlineCitation>"
            f"<PMID>{a['pmid']}</PMID>"
            f"<Article>"
            f"<ArticleTitle>{a['title']}</ArticleTitle>"
            f"<AuthorList>{authors}</AuthorList>"
            f"<Abstract><AbstractText>{a.get('abstract', '')}</AbstractText></Abstract>"
            f"</Article>"
            f"</MedlineCitation></PubmedArticle>"
        )
    return f"<PubmedArticleSet>{''.join(parts)}</PubmedArticleSet>".encode()


class TestPubMedSearchTool:
    def test_import(self):
        from synapsekit import PubMedSearchTool

        assert PubMedSearchTool is not None

    async def test_search_returns_results(self):
        from synapsekit.agents.tools.pubmed_search import PubMedSearchTool

        esearch_resp = MagicMock()
        esearch_resp.read.return_value = _make_esearch_xml(["12345"])
        esearch_resp.__enter__ = MagicMock(return_value=esearch_resp)
        esearch_resp.__exit__ = MagicMock(return_value=False)

        efetch_resp = MagicMock()
        efetch_resp.read.return_value = _make_efetch_xml(
            [
                {
                    "pmid": "12345",
                    "title": "CRISPR Study",
                    "authors": ["Smith John"],
                    "abstract": "A study.",
                }
            ]
        )
        efetch_resp.__enter__ = MagicMock(return_value=efetch_resp)
        efetch_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", side_effect=[esearch_resp, efetch_resp]):
            tool = PubMedSearchTool()
            result = await tool.run(query="CRISPR")

        assert "CRISPR Study" in result.output
        assert "12345" in result.output
        assert not result.is_error

    async def test_empty_results(self):
        from synapsekit.agents.tools.pubmed_search import PubMedSearchTool

        resp = MagicMock()
        resp.read.return_value = b"<eSearchResult><IdList></IdList></eSearchResult>"
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=resp):
            tool = PubMedSearchTool()
            result = await tool.run(query="xyznonexistent")

        assert result.output == "No results found."

    async def test_no_query_error(self):
        from synapsekit.agents.tools.pubmed_search import PubMedSearchTool

        tool = PubMedSearchTool()
        result = await tool.run()

        assert result.is_error
        assert "No search query" in result.error


# ------------------------------------------------------------------ #
# Feature 3: GitHubAPITool
# ------------------------------------------------------------------ #


class TestGitHubAPITool:
    def test_import(self):
        from synapsekit import GitHubAPITool

        assert GitHubAPITool is not None

    async def test_search_repos(self):
        from synapsekit.agents.tools.github_api import GitHubAPITool

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {
                "items": [
                    {
                        "full_name": "org/repo",
                        "stargazers_count": 100,
                        "description": "A great repo",
                        "html_url": "https://github.com/org/repo",
                    }
                ]
            }
        ).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            tool = GitHubAPITool(token="test-token")
            result = await tool.run(action="search_repos", query="langchain")

        assert "org/repo" in result.output
        assert not result.is_error

    async def test_get_repo(self):
        from synapsekit.agents.tools.github_api import GitHubAPITool

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {
                "full_name": "org/repo",
                "stargazers_count": 50,
                "forks_count": 10,
                "description": "Test",
                "language": "Python",
                "html_url": "https://github.com/org/repo",
            }
        ).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            tool = GitHubAPITool(token="test-token")
            result = await tool.run(action="get_repo", owner="org", repo="repo")

        assert "org/repo" in result.output
        assert "Python" in result.output

    async def test_search_issues(self):
        from synapsekit.agents.tools.github_api import GitHubAPITool

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {
                "items": [
                    {
                        "title": "Bug fix",
                        "number": 42,
                        "state": "open",
                        "html_url": "https://github.com/org/repo/issues/42",
                    }
                ]
            }
        ).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            tool = GitHubAPITool(token="test-token")
            result = await tool.run(action="search_issues", query="bug")

        assert "Bug fix" in result.output
        assert "#42" in result.output

    async def test_get_issue(self):
        from synapsekit.agents.tools.github_api import GitHubAPITool

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {
                "title": "Bug fix",
                "number": 42,
                "state": "open",
                "user": {"login": "dev"},
                "html_url": "https://github.com/org/repo/issues/42",
                "body": "Fix the thing",
            }
        ).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            tool = GitHubAPITool(token="test-token")
            result = await tool.run(action="get_issue", owner="org", repo="repo", issue_number=42)

        assert "Bug fix" in result.output
        assert "dev" in result.output

    async def test_missing_action(self):
        from synapsekit.agents.tools.github_api import GitHubAPITool

        tool = GitHubAPITool()
        result = await tool.run()
        assert result.is_error
        assert "No action" in result.error

    async def test_unknown_action(self):
        from synapsekit.agents.tools.github_api import GitHubAPITool

        tool = GitHubAPITool()
        result = await tool.run(action="delete_everything")
        assert result.is_error
        assert "Unknown action" in result.error

    async def test_missing_params_get_repo(self):
        from synapsekit.agents.tools.github_api import GitHubAPITool

        tool = GitHubAPITool(token="test")
        result = await tool.run(action="get_repo", owner="org")
        assert result.is_error
        assert "required" in result.error.lower()


# ------------------------------------------------------------------ #
# Feature 4: EmailTool
# ------------------------------------------------------------------ #


class TestEmailTool:
    def test_import(self):
        from synapsekit import EmailTool

        assert EmailTool is not None

    async def test_successful_send(self):
        from synapsekit.agents.tools.email_tool import EmailTool

        mock_smtp = MagicMock()
        mock_smtp.__enter__ = MagicMock(return_value=mock_smtp)
        mock_smtp.__exit__ = MagicMock(return_value=False)

        with patch("smtplib.SMTP", return_value=mock_smtp):
            tool = EmailTool(
                smtp_host="smtp.test.com",
                smtp_port=587,
                smtp_user="user@test.com",
                smtp_password="pass",
            )
            result = await tool.run(to="bob@example.com", subject="Hi", body="Hello")

        assert "sent successfully" in result.output
        mock_smtp.starttls.assert_called_once()
        mock_smtp.login.assert_called_once_with("user@test.com", "pass")
        mock_smtp.send_message.assert_called_once()

    async def test_missing_params(self):
        from synapsekit.agents.tools.email_tool import EmailTool

        tool = EmailTool(smtp_host="h", smtp_user="u", smtp_password="p")
        result = await tool.run(to="bob@test.com", subject="Hi")
        assert result.is_error
        assert "required" in result.error.lower()

    async def test_missing_smtp_config(self):
        from synapsekit.agents.tools.email_tool import EmailTool

        with patch.dict("os.environ", {}, clear=True):
            tool = EmailTool()
            result = await tool.run(to="bob@test.com", subject="Hi", body="Hello")

        assert result.is_error
        assert "SMTP not configured" in result.error

    async def test_from_addr_override(self):
        from synapsekit.agents.tools.email_tool import EmailTool

        mock_smtp = MagicMock()
        mock_smtp.__enter__ = MagicMock(return_value=mock_smtp)
        mock_smtp.__exit__ = MagicMock(return_value=False)

        with patch("smtplib.SMTP", return_value=mock_smtp):
            tool = EmailTool(
                smtp_host="h",
                smtp_user="u",
                smtp_password="p",
                from_addr="custom@test.com",
            )
            result = await tool.run(to="bob@test.com", subject="Hi", body="Hello")

        assert not result.is_error
        sent_msg = mock_smtp.send_message.call_args[0][0]
        assert sent_msg["From"] == "custom@test.com"


# ------------------------------------------------------------------ #
# Feature 5: YouTubeSearchTool
# ------------------------------------------------------------------ #


class TestYouTubeSearchTool:
    def test_import(self):
        from synapsekit import YouTubeSearchTool

        assert YouTubeSearchTool is not None

    async def test_search_returns_results(self):
        from synapsekit.agents.tools.youtube_search import YouTubeSearchTool

        mock_module = ModuleType("youtubesearchpython")
        mock_search_instance = MagicMock()
        mock_search_instance.result.return_value = {
            "result": [
                {
                    "title": "Python Tutorial",
                    "channel": {"name": "Tech Channel"},
                    "duration": "10:30",
                    "link": "https://youtube.com/watch?v=abc",
                    "viewCount": {"short": "1.2M views"},
                }
            ]
        }
        mock_videos_search = MagicMock(return_value=mock_search_instance)
        mock_module.VideosSearch = mock_videos_search

        with patch.dict(sys.modules, {"youtubesearchpython": mock_module}):
            tool = YouTubeSearchTool()
            result = await tool.run(query="python tutorial")

        assert "Python Tutorial" in result.output
        assert "Tech Channel" in result.output
        assert not result.is_error

    async def test_empty_results(self):
        from synapsekit.agents.tools.youtube_search import YouTubeSearchTool

        mock_module = ModuleType("youtubesearchpython")
        mock_search_instance = MagicMock()
        mock_search_instance.result.return_value = {"result": []}
        mock_module.VideosSearch = MagicMock(return_value=mock_search_instance)

        with patch.dict(sys.modules, {"youtubesearchpython": mock_module}):
            tool = YouTubeSearchTool()
            result = await tool.run(query="xyznonexistent")

        assert result.output == "No results found."

    async def test_no_query_error(self):
        from synapsekit.agents.tools.youtube_search import YouTubeSearchTool

        tool = YouTubeSearchTool()
        result = await tool.run()
        assert result.is_error
        assert "No search query" in result.error

    async def test_missing_dependency(self):
        from synapsekit.agents.tools.youtube_search import YouTubeSearchTool

        with patch.dict(sys.modules, {"youtubesearchpython": None}):
            tool = YouTubeSearchTool()
            with pytest.raises(ImportError, match="youtube-search-python"):
                await tool.run(query="test")


# ------------------------------------------------------------------ #
# Feature 6: ExecutionTrace
# ------------------------------------------------------------------ #


class TestExecutionTrace:
    def test_import(self):
        from synapsekit import ExecutionTrace, TraceEntry

        assert ExecutionTrace is not None
        assert TraceEntry is not None

    async def test_trace_captures_events(self):
        from synapsekit.graph.streaming import EventHooks, GraphEvent
        from synapsekit.graph.trace import ExecutionTrace

        trace = ExecutionTrace()
        hooks = trace.hook(EventHooks())

        await hooks.emit(GraphEvent(event_type="node_start", node="a"))
        await hooks.emit(GraphEvent(event_type="node_complete", node="a"))

        assert len(trace.entries) == 2
        assert trace.entries[0].event_type == "node_start"
        assert trace.entries[1].event_type == "node_complete"

    async def test_duration_calculation(self):
        from synapsekit.graph.streaming import EventHooks, GraphEvent
        from synapsekit.graph.trace import ExecutionTrace

        trace = ExecutionTrace()
        hooks = trace.hook(EventHooks())

        await hooks.emit(GraphEvent(event_type="node_start", node="slow"))
        time.sleep(0.01)  # 10ms
        await hooks.emit(GraphEvent(event_type="node_complete", node="slow"))

        durations = trace.node_durations
        assert "slow" in durations
        assert durations["slow"] >= 5  # at least 5ms

    async def test_summary_format(self):
        from synapsekit.graph.streaming import EventHooks, GraphEvent
        from synapsekit.graph.trace import ExecutionTrace

        trace = ExecutionTrace()
        hooks = trace.hook(EventHooks())

        await hooks.emit(GraphEvent(event_type="node_start", node="x"))
        await hooks.emit(GraphEvent(event_type="node_complete", node="x"))

        summary = trace.summary()
        assert "Execution trace" in summary
        assert "2 events" in summary
        assert "node_start" in summary
        assert "[x]" in summary

    async def test_to_dict_serialization(self):
        from synapsekit.graph.streaming import EventHooks, GraphEvent
        from synapsekit.graph.trace import ExecutionTrace

        trace = ExecutionTrace()
        hooks = trace.hook(EventHooks())

        await hooks.emit(GraphEvent(event_type="node_start", node="a"))
        await hooks.emit(GraphEvent(event_type="node_complete", node="a"))

        data = trace.to_dict()
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["event_type"] == "node_start"
        assert "timestamp" in data[0]
        # Should be JSON-serializable
        json.dumps(data)

    def test_empty_trace(self):
        from synapsekit.graph.trace import ExecutionTrace

        trace = ExecutionTrace()
        assert trace.entries == []
        assert trace.total_duration_ms == 0.0
        assert trace.node_durations == {}
        assert trace.summary() == "No events recorded."
        assert trace.to_dict() == []

    async def test_wave_and_error_events(self):
        from synapsekit.graph.streaming import EventHooks, GraphEvent
        from synapsekit.graph.trace import ExecutionTrace

        trace = ExecutionTrace()
        hooks = trace.hook(EventHooks())

        await hooks.emit(GraphEvent(event_type="wave_start"))
        await hooks.emit(GraphEvent(event_type="error", node="bad", data={"msg": "oops"}))
        await hooks.emit(GraphEvent(event_type="wave_complete"))

        assert len(trace.entries) == 3
        assert trace.entries[1].event_type == "error"
        assert trace.entries[1].data == {"msg": "oops"}


# ------------------------------------------------------------------ #
# Feature 7: WebSocket Streaming
# ------------------------------------------------------------------ #


class TestWebSocketStreaming:
    def test_to_ws(self):
        from synapsekit.graph.streaming import GraphEvent

        event = GraphEvent(event_type="node_complete", node="a", state={"x": 1})
        ws_msg = event.to_ws()
        parsed = json.loads(ws_msg)
        assert parsed["event"] == "node_complete"
        assert parsed["node"] == "a"
        assert parsed["state"] == {"x": 1}

    async def test_ws_stream_sends_events(self):
        from synapsekit.graph.streaming import ws_stream

        mock_ws = AsyncMock()
        mock_ws.send_text = AsyncMock()

        mock_graph = AsyncMock()
        mock_graph.run = AsyncMock(return_value={"output": "done"})

        result = await ws_stream(mock_graph, {"input": "hello"}, mock_ws)

        assert result == {"output": "done"}
        # Should have sent the "done" event at minimum
        mock_ws.send_text.assert_called()
        last_call = mock_ws.send_text.call_args[0][0]
        parsed = json.loads(last_call)
        assert parsed["event"] == "done"

    async def test_ws_stream_fallback_to_send(self):
        from synapsekit.graph.streaming import ws_stream

        mock_ws = AsyncMock(spec=[])  # no send_text
        mock_ws.send = AsyncMock()

        mock_graph = AsyncMock()
        mock_graph.run = AsyncMock(return_value={"result": "ok"})

        result = await ws_stream(mock_graph, {}, mock_ws)

        assert result == {"result": "ok"}
        mock_ws.send.assert_called()

    async def test_ws_stream_with_user_hooks(self):
        from synapsekit.graph.streaming import EventHooks, ws_stream

        mock_ws = AsyncMock()
        mock_ws.send_text = AsyncMock()

        user_events = []
        user_hooks = EventHooks()
        user_hooks.on_node_start(lambda e: user_events.append(e.event_type))

        mock_graph = AsyncMock()
        mock_graph.run = AsyncMock(return_value={})

        await ws_stream(mock_graph, {}, mock_ws, hooks=user_hooks)

        # User hooks should have been merged
        mock_graph.run.assert_called_once()

    async def test_ws_stream_error_in_graph(self):
        from synapsekit.graph.streaming import ws_stream

        mock_ws = AsyncMock()
        mock_ws.send_text = AsyncMock()

        mock_graph = AsyncMock()
        mock_graph.run = AsyncMock(side_effect=RuntimeError("graph failed"))

        with pytest.raises(RuntimeError, match="graph failed"):
            await ws_stream(mock_graph, {}, mock_ws)


# ------------------------------------------------------------------ #
# Export tests
# ------------------------------------------------------------------ #


class TestExports:
    def test_all_new_tools_in_top_level(self):
        import synapsekit

        for name in [
            "VectorSearchTool",
            "PubMedSearchTool",
            "GitHubAPITool",
            "EmailTool",
            "YouTubeSearchTool",
        ]:
            assert hasattr(synapsekit, name), f"{name} not exported from synapsekit"
            assert name in synapsekit.__all__, f"{name} not in synapsekit.__all__"

    def test_graph_exports(self):
        import synapsekit

        for name in ["ExecutionTrace", "TraceEntry", "ws_stream"]:
            assert hasattr(synapsekit, name), f"{name} not exported from synapsekit"
            assert name in synapsekit.__all__, f"{name} not in synapsekit.__all__"

    def test_tool_schemas(self):
        from synapsekit import (
            EmailTool,
            GitHubAPITool,
            PubMedSearchTool,
            VectorSearchTool,
            YouTubeSearchTool,
        )

        for cls, kwargs in [
            (VectorSearchTool, {"retriever": AsyncMock()}),
            (PubMedSearchTool, {}),
            (GitHubAPITool, {}),
            (EmailTool, {}),
            (YouTubeSearchTool, {}),
        ]:
            tool = cls(**kwargs)
            schema = tool.schema()
            assert schema["type"] == "function"
            assert "name" in schema["function"]

            anthropic = tool.anthropic_schema()
            assert "name" in anthropic
            assert "input_schema" in anthropic
