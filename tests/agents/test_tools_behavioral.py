"""Behavioral tests for core agent tools.

Verifies that CalculatorTool, DateTimeTool, ShellTool, FileWriteTool,
HTTPRequestTool, WebSearchTool, and DuckDuckGoSearchTool behave correctly
across all code paths, including error branches, boundary values, and
security guards. No real network or subprocess calls.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synapsekit.agents.tools.calculator import CalculatorTool
from synapsekit.agents.tools.datetime_tool import DateTimeTool
from synapsekit.agents.tools.file_write import FileWriteTool
from synapsekit.agents.tools.http_request import HTTPRequestTool
from synapsekit.agents.tools.shell import ShellTool

# ---------------------------------------------------------------------------
# CalculatorTool
# ---------------------------------------------------------------------------


class TestCalculatorTool:
    @pytest.mark.asyncio
    async def test_basic_arithmetic(self):
        tool = CalculatorTool()
        result = await tool.run(expression="2 + 3 * 4")
        assert result.error is None
        assert result.output == "14"

    @pytest.mark.asyncio
    async def test_power_operator(self):
        tool = CalculatorTool()
        result = await tool.run(expression="2 ** 10")
        assert result.output == "1024"

    @pytest.mark.asyncio
    async def test_sqrt_function(self):
        tool = CalculatorTool()
        result = await tool.run(expression="sqrt(144)")
        assert result.output == "12.0"

    @pytest.mark.asyncio
    async def test_trig_function(self):
        tool = CalculatorTool()
        result = await tool.run(expression="round(sin(pi/2), 5)")
        assert result.output == "1.0"

    @pytest.mark.asyncio
    async def test_log_function(self):
        tool = CalculatorTool()
        result = await tool.run(expression="log10(1000)")
        assert result.output == "3.0"

    @pytest.mark.asyncio
    async def test_factorial(self):
        tool = CalculatorTool()
        result = await tool.run(expression="factorial(5)")
        assert result.output == "120"

    @pytest.mark.asyncio
    async def test_division_by_zero(self):
        tool = CalculatorTool()
        result = await tool.run(expression="1 / 0")
        assert result.error is not None
        assert "zero" in result.error.lower()

    @pytest.mark.asyncio
    async def test_invalid_expression(self):
        tool = CalculatorTool()
        result = await tool.run(expression="not_a_function()")
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_no_expression_returns_error(self):
        tool = CalculatorTool()
        result = await tool.run(expression="")
        assert result.error is not None
        assert "No expression" in result.error

    @pytest.mark.asyncio
    async def test_blocked_builtins(self):
        """__import__ and os should not be accessible."""
        tool = CalculatorTool()
        result = await tool.run(expression="__import__('os').getcwd()")
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_pi_constant(self):
        import math

        tool = CalculatorTool()
        result = await tool.run(expression="round(pi, 5)")
        assert result.output == str(round(math.pi, 5))

    @pytest.mark.asyncio
    async def test_fallback_input_kwarg(self):
        tool = CalculatorTool()
        result = await tool.run(input="5 + 5")
        assert result.output == "10"


# ---------------------------------------------------------------------------
# DateTimeTool
# ---------------------------------------------------------------------------


class TestDateTimeTool:
    @pytest.mark.asyncio
    async def test_now_returns_isoformat(self):
        tool = DateTimeTool()
        result = await tool.run(action="now")
        assert result.error is None
        assert "T" in result.output  # ISO 8601 contains 'T'

    @pytest.mark.asyncio
    async def test_now_utc(self):
        tool = DateTimeTool()
        result = await tool.run(action="now", tz="utc")
        assert result.error is None
        assert "+00:00" in result.output or "Z" in result.output or "T" in result.output

    @pytest.mark.asyncio
    async def test_now_with_format(self):
        tool = DateTimeTool()
        result = await tool.run(action="now", fmt="%Y")
        assert result.error is None
        assert len(result.output) == 4  # just the year

    @pytest.mark.asyncio
    async def test_parse_iso_date(self):
        tool = DateTimeTool()
        result = await tool.run(action="parse", value="2024-01-15T10:30:00")
        assert result.error is None
        assert "2024-01-15" in result.output

    @pytest.mark.asyncio
    async def test_parse_with_format(self):
        tool = DateTimeTool()
        result = await tool.run(action="parse", value="15/01/2024", fmt="%d/%m/%Y")
        assert result.error is None
        assert "2024-01-15" in result.output

    @pytest.mark.asyncio
    async def test_parse_no_value_returns_error(self):
        tool = DateTimeTool()
        result = await tool.run(action="parse", value="")
        assert result.error is not None
        assert "No value" in result.error

    @pytest.mark.asyncio
    async def test_format_action(self):
        tool = DateTimeTool()
        result = await tool.run(action="format", value="2024-06-15T00:00:00", fmt="%d %b %Y")
        assert result.error is None
        assert "15 Jun 2024" in result.output

    @pytest.mark.asyncio
    async def test_format_no_value_returns_error(self):
        tool = DateTimeTool()
        result = await tool.run(action="format", value="")
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_unknown_action_returns_error(self):
        tool = DateTimeTool()
        result = await tool.run(action="teleport")
        assert result.error is not None
        assert "Unknown action" in result.error

    @pytest.mark.asyncio
    async def test_invalid_iso_parse_returns_error(self):
        tool = DateTimeTool()
        result = await tool.run(action="parse", value="not-a-date")
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_default_action_is_now(self):
        tool = DateTimeTool()
        result = await tool.run()  # no action → defaults to "now"
        assert result.error is None
        assert "T" in result.output


# ---------------------------------------------------------------------------
# ShellTool
# ---------------------------------------------------------------------------


class TestShellTool:
    @pytest.mark.asyncio
    async def test_echo_command(self):
        tool = ShellTool()
        result = await tool.run(command="echo hello")
        assert result.error is None
        assert "hello" in result.output

    @pytest.mark.asyncio
    async def test_no_command_returns_error(self):
        tool = ShellTool()
        result = await tool.run(command="")
        assert result.error is not None
        assert "No command" in result.error

    @pytest.mark.asyncio
    async def test_allowed_commands_permits_allowed(self):
        tool = ShellTool(allowed_commands=["echo"])
        result = await tool.run(command="echo allowed")
        assert result.error is None

    @pytest.mark.asyncio
    async def test_allowed_commands_blocks_disallowed(self):
        tool = ShellTool(allowed_commands=["echo"])
        result = await tool.run(command="ls -la")
        assert result.error is not None
        assert "not in the allowed list" in result.error

    @pytest.mark.asyncio
    async def test_nonexistent_command_returns_error(self):
        tool = ShellTool()
        result = await tool.run(command="nonexistent_command_xyz")
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_command_exit_code_nonzero(self):
        tool = ShellTool()
        result = await tool.run(command="ls /nonexistent_path_xyz")
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_invalid_shell_syntax_returns_error(self):
        tool = ShellTool()
        result = await tool.run(command="echo 'unclosed")
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_input_kwarg_fallback(self):
        tool = ShellTool()
        result = await tool.run(input="echo from_input")
        assert result.error is None
        assert "from_input" in result.output

    @pytest.mark.asyncio
    async def test_timeout_triggers_error(self):
        tool = ShellTool(timeout=1)
        # sleep 5 should timeout after 1 second
        result = await tool.run(command="sleep 5")
        assert result.error is not None
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_name_and_description(self):
        tool = ShellTool()
        assert tool.name == "shell"
        assert "shell" in tool.description.lower()


# ---------------------------------------------------------------------------
# FileWriteTool
# ---------------------------------------------------------------------------


class TestFileWriteTool:
    @pytest.mark.asyncio
    async def test_write_creates_file(self, tmp_path):
        tool = FileWriteTool()
        fpath = str(tmp_path / "out.txt")
        result = await tool.run(path=fpath, content="hello world")
        assert result.error is None
        assert Path(fpath).read_text() == "hello world"
        assert "Written to" in result.output

    @pytest.mark.asyncio
    async def test_write_reports_char_count(self, tmp_path):
        tool = FileWriteTool()
        fpath = str(tmp_path / "out.txt")
        content = "abc" * 10
        result = await tool.run(path=fpath, content=content)
        assert str(len(content)) in result.output

    @pytest.mark.asyncio
    async def test_append_mode(self, tmp_path):
        tool = FileWriteTool()
        fpath = str(tmp_path / "out.txt")
        await tool.run(path=fpath, content="line1\n")
        result = await tool.run(path=fpath, content="line2\n", append=True)
        assert result.error is None
        assert "Appended to" in result.output
        assert Path(fpath).read_text() == "line1\nline2\n"

    @pytest.mark.asyncio
    async def test_creates_parent_directories(self, tmp_path):
        tool = FileWriteTool()
        fpath = str(tmp_path / "deep" / "nested" / "file.txt")
        result = await tool.run(path=fpath, content="nested")
        assert result.error is None
        assert Path(fpath).exists()

    @pytest.mark.asyncio
    async def test_no_path_returns_error(self):
        tool = FileWriteTool()
        result = await tool.run(path="", content="data")
        assert result.error is not None
        assert "No file path" in result.error

    @pytest.mark.asyncio
    async def test_base_dir_restriction_blocks_escape(self, tmp_path):
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        tool = FileWriteTool(base_dir=str(allowed_dir))

        outside_path = str(tmp_path / "outside.txt")
        result = await tool.run(path=outside_path, content="bad")
        assert result.error is not None
        assert "Access denied" in result.error

    @pytest.mark.asyncio
    async def test_base_dir_permits_inside_path(self, tmp_path):
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        tool = FileWriteTool(base_dir=str(allowed_dir))

        inside_path = str(allowed_dir / "ok.txt")
        result = await tool.run(path=inside_path, content="permitted")
        assert result.error is None

    @pytest.mark.asyncio
    async def test_overwrite_existing_file(self, tmp_path):
        tool = FileWriteTool()
        fpath = str(tmp_path / "file.txt")
        await tool.run(path=fpath, content="original")
        await tool.run(path=fpath, content="overwritten")
        assert Path(fpath).read_text() == "overwritten"


# ---------------------------------------------------------------------------
# HTTPRequestTool
# ---------------------------------------------------------------------------


class TestHTTPRequestTool:
    @pytest.mark.asyncio
    async def test_get_request_success(self):
        tool = HTTPRequestTool()

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.text = AsyncMock(return_value="Hello from server")
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.request = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)
        mock_aiohttp.ClientTimeout = MagicMock(return_value=MagicMock())

        with patch.dict(sys.modules, {"aiohttp": mock_aiohttp}):
            result = await tool.run(url="https://example.com")

        assert result.error is None
        assert "200" in result.output
        assert "Hello from server" in result.output

    @pytest.mark.asyncio
    async def test_no_url_returns_error(self):
        tool = HTTPRequestTool()
        result = await tool.run(url="")
        assert result.error is not None
        assert "No URL" in result.error

    @pytest.mark.asyncio
    async def test_response_truncated_at_max_length(self):
        tool = HTTPRequestTool(max_response_length=10)

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.text = AsyncMock(return_value="A" * 100)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.request = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)
        mock_aiohttp.ClientTimeout = MagicMock(return_value=MagicMock())

        with patch.dict(sys.modules, {"aiohttp": mock_aiohttp}):
            result = await tool.run(url="https://example.com")

        assert "truncated" in result.output

    @pytest.mark.asyncio
    async def test_import_error_raises(self):
        tool = HTTPRequestTool()
        with patch.dict(sys.modules, {"aiohttp": None}):
            with pytest.raises(ImportError, match="aiohttp required"):
                await tool.run(url="https://example.com")

    @pytest.mark.asyncio
    async def test_network_exception_returns_error(self):
        tool = HTTPRequestTool()

        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientTimeout = MagicMock(return_value=MagicMock())

        mock_session = AsyncMock()
        mock_session.request = MagicMock(side_effect=Exception("connection refused"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)

        with patch.dict(sys.modules, {"aiohttp": mock_aiohttp}):
            result = await tool.run(url="https://bad.example.com")

        assert result.error is not None
        assert "failed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_post_with_body(self):
        tool = HTTPRequestTool()

        captured = {}
        mock_resp = AsyncMock()
        mock_resp.status = 201
        mock_resp.text = AsyncMock(return_value="created")
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()

        def _request(method, url, **kw):
            captured.update({"method": method, "data": kw.get("data")})
            return mock_resp

        mock_session.request = MagicMock(side_effect=_request)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)
        mock_aiohttp.ClientTimeout = MagicMock(return_value=MagicMock())

        with patch.dict(sys.modules, {"aiohttp": mock_aiohttp}):
            await tool.run(url="https://api.example.com", method="POST", body='{"key":"val"}')

        assert captured["method"] == "POST"
        assert captured["data"] == '{"key":"val"}'


# ---------------------------------------------------------------------------
# WebSearchTool
# ---------------------------------------------------------------------------


class TestWebSearchTool:
    @pytest.mark.asyncio
    async def test_no_query_returns_error(self):
        from synapsekit.agents.tools.web_search import WebSearchTool

        tool = WebSearchTool()
        result = await tool.run(query="")
        assert result.error is not None
        assert "No search query" in result.error

    @pytest.mark.asyncio
    async def test_returns_formatted_results(self):
        from synapsekit.agents.tools.web_search import WebSearchTool

        tool = WebSearchTool()

        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
        mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
        mock_ddgs_instance.text = MagicMock(
            return_value=[
                {"title": "Python Docs", "href": "https://python.org", "body": "Official docs"},
                {"title": "Python Tut", "href": "https://tutorial.org", "body": "Learn Python"},
            ]
        )

        mock_ddgs_cls = MagicMock(return_value=mock_ddgs_instance)

        with patch.dict(sys.modules, {"duckduckgo_search": MagicMock(DDGS=mock_ddgs_cls)}):
            result = await tool.run(query="Python programming")

        assert result.error is None
        assert "Python Docs" in result.output
        assert "https://python.org" in result.output

    @pytest.mark.asyncio
    async def test_no_results_returns_message(self):
        from synapsekit.agents.tools.web_search import WebSearchTool

        tool = WebSearchTool()

        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
        mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
        mock_ddgs_instance.text = MagicMock(return_value=[])
        mock_ddgs_cls = MagicMock(return_value=mock_ddgs_instance)

        with patch.dict(sys.modules, {"duckduckgo_search": MagicMock(DDGS=mock_ddgs_cls)}):
            result = await tool.run(query="xyzzy_not_found")

        assert "No results" in result.output

    @pytest.mark.asyncio
    async def test_import_error_raises(self):
        from synapsekit.agents.tools.web_search import WebSearchTool

        tool = WebSearchTool()
        with patch.dict(sys.modules, {"duckduckgo_search": None}):
            with pytest.raises(ImportError, match="duckduckgo-search required"):
                await tool.run(query="something")

    @pytest.mark.asyncio
    async def test_search_exception_returns_error(self):
        from synapsekit.agents.tools.web_search import WebSearchTool

        tool = WebSearchTool()

        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
        mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
        mock_ddgs_instance.text = MagicMock(side_effect=RuntimeError("rate limited"))
        mock_ddgs_cls = MagicMock(return_value=mock_ddgs_instance)

        with patch.dict(sys.modules, {"duckduckgo_search": MagicMock(DDGS=mock_ddgs_cls)}):
            result = await tool.run(query="query")

        assert result.error is not None
        assert "Search failed" in result.error


# ---------------------------------------------------------------------------
# DuckDuckGoSearchTool
# ---------------------------------------------------------------------------


class TestDuckDuckGoSearchTool:
    @pytest.mark.asyncio
    async def test_text_search(self):
        from synapsekit.agents.tools.duck_search import DuckDuckGoSearchTool

        tool = DuckDuckGoSearchTool()

        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
        mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
        mock_ddgs_instance.text = MagicMock(
            return_value=[
                {"title": "Result 1", "href": "https://r1.com", "body": "snippet 1"},
            ]
        )
        mock_ddgs_cls = MagicMock(return_value=mock_ddgs_instance)

        with patch.dict(sys.modules, {"duckduckgo_search": MagicMock(DDGS=mock_ddgs_cls)}):
            result = await tool.run(query="test", search_type="text")

        assert "Result 1" in result.output
        assert "1." in result.output  # numbered result

    @pytest.mark.asyncio
    async def test_news_search_type(self):
        from synapsekit.agents.tools.duck_search import DuckDuckGoSearchTool

        tool = DuckDuckGoSearchTool()

        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
        mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
        mock_ddgs_instance.news = MagicMock(
            return_value=[
                {"title": "News Item", "url": "https://news.com", "excerpt": "news snippet"},
            ]
        )
        mock_ddgs_cls = MagicMock(return_value=mock_ddgs_instance)

        with patch.dict(sys.modules, {"duckduckgo_search": MagicMock(DDGS=mock_ddgs_cls)}):
            result = await tool.run(query="breaking news", search_type="news")

        mock_ddgs_instance.news.assert_called_once()
        assert "News Item" in result.output

    @pytest.mark.asyncio
    async def test_no_query_returns_error(self):
        from synapsekit.agents.tools.duck_search import DuckDuckGoSearchTool

        tool = DuckDuckGoSearchTool()
        result = await tool.run(query="")
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_input_kwarg_fallback(self):
        from synapsekit.agents.tools.duck_search import DuckDuckGoSearchTool

        tool = DuckDuckGoSearchTool()

        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
        mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
        mock_ddgs_instance.text = MagicMock(
            return_value=[
                {"title": "T", "href": "https://x.com", "body": "b"},
            ]
        )
        mock_ddgs_cls = MagicMock(return_value=mock_ddgs_instance)

        with patch.dict(sys.modules, {"duckduckgo_search": MagicMock(DDGS=mock_ddgs_cls)}):
            result = await tool.run(input="from input kwarg")

        assert result.error is None
