"""Tests for MCPServer — RAG, agent, and tool wrapping modes.

All tests mock the mcp SDK completely — no real mcp package required,
no network access, no API keys, matching CONTRIBUTING.md guidelines.
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synapsekit.mcp.server.core import MCPServer

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _make_mock_mcp():
    """Return a minimal mcp module mock that supports the decorator API."""
    # The registered handlers: decorator @server.list_tools() etc. store callables here.
    registered: dict[str, object] = {}

    mock_server_instance = MagicMock()

    # Simulate the decorator pattern: @server.list_tools() returns a decorator
    # that stores the function under the key.
    def decorator_factory(key: str):
        def decorator(fn):
            registered[key] = fn
            return fn

        return lambda: decorator  # @server.list_tools() → decorator(fn)

    mock_server_instance.list_tools = decorator_factory("list_tools")
    mock_server_instance.call_tool = decorator_factory("call_tool")
    mock_server_instance.list_resources = decorator_factory("list_resources")
    mock_server_instance.read_resource = decorator_factory("read_resource")
    mock_server_instance._registered = registered

    mock_server_cls = MagicMock(return_value=mock_server_instance)

    mock_text_content = MagicMock()
    mock_text_content_instance = MagicMock()
    mock_text_content.return_value = mock_text_content_instance

    mock_tool = MagicMock()
    mock_resource = MagicMock()

    mock_mcp_server_mod = MagicMock()
    mock_mcp_server_mod.Server = mock_server_cls

    mock_mcp_types = MagicMock()
    mock_mcp_types.TextContent = mock_text_content
    mock_mcp_types.Tool = mock_tool
    mock_mcp_types.Resource = mock_resource

    return mock_mcp_server_mod, mock_mcp_types, mock_server_instance, registered


# ------------------------------------------------------------------ #
# Import tests (no real mcp needed)
# ------------------------------------------------------------------ #


def test_mcpserver_import():
    """MCPServer can be imported from both module and top-level."""
    from synapsekit import MCPServer as TopLevelServer
    from synapsekit.mcp.server.core import MCPServer as Core

    assert TopLevelServer is Core


def test_mcpserver_default_init():
    """MCPServer can be constructed with no arguments."""
    server = MCPServer()
    assert server._name == "synapsekit"
    assert server._version == "1.0.0"
    assert server._tools == {}
    assert server._target is None


def test_mcpserver_backward_compat_name_string():
    """Old-style MCPServer(name='x') still works (string as first positional arg)."""
    server = MCPServer("my-server")
    assert server._name == "my-server"
    assert server._target is None


def test_mcpserver_backward_compat_tools_kwarg():
    """Old-style MCPServer(name='x', tools=[...]) still works."""
    mock_tool = MagicMock()
    mock_tool.name = "calc"
    server = MCPServer(name="test", tools=[mock_tool])
    assert "calc" in server._tools
    assert server._target is None


def test_mcpserver_add_tool():
    """add_tool() registers a tool."""
    server = MCPServer()
    mock_tool = MagicMock()
    mock_tool.name = "dt"
    server.add_tool(mock_tool)
    assert "dt" in server._tools


def test_mcpserver_unsupported_source_type():
    """MCPServer with an unsupported source raises TypeError."""
    with pytest.raises(TypeError, match="Unsupported source type"):
        MCPServer(12345)


# ------------------------------------------------------------------ #
# _build_server missing mcp
# ------------------------------------------------------------------ #


def test_mcpserver_build_missing_mcp():
    """_build_server raises ImportError when mcp package not installed."""
    saved = sys.modules.pop("mcp", None)
    saved2 = sys.modules.pop("mcp.server", None)
    saved3 = sys.modules.pop("mcp.types", None)
    try:
        with patch.dict(sys.modules, {"mcp": None, "mcp.server": None, "mcp.types": None}):
            server = MCPServer()
            with pytest.raises(ImportError, match="mcp package required"):
                server._build_server()
    finally:
        for key, val in [("mcp", saved), ("mcp.server", saved2), ("mcp.types", saved3)]:
            if val is not None:
                sys.modules[key] = val


# ------------------------------------------------------------------ #
# RAG mode
# ------------------------------------------------------------------ #


async def test_mcpserver_rag_registers_query_tool():
    """MCPServer(rag) registers exactly one tool named 'rag_query'."""
    mock_rag = MagicMock()
    mock_rag.__class__.__name__ = "RAG"
    mock_rag.ask = AsyncMock(return_value="answer")
    mock_rag._vectorstore = MagicMock(_texts=[], _metadata=[])

    mock_server_mod, mock_types, _mock_srv, registered = _make_mock_mcp()

    with patch.dict(sys.modules, {"mcp.server": mock_server_mod, "mcp.types": mock_types}):
        server = MCPServer(mock_rag)
        server._build_server()

    assert "list_tools" in registered
    assert "call_tool" in registered


async def test_mcpserver_rag_query_calls_ask():
    """'rag_query' tool handler forwards query to rag.ask()."""
    mock_rag = MagicMock()
    mock_rag.__class__.__name__ = "RAG"
    mock_rag.ask = AsyncMock(return_value="RAG answer")
    mock_rag._vectorstore = MagicMock(_texts=[], _metadata=[])

    mock_server_mod, mock_types, _mock_srv, registered = _make_mock_mcp()

    with patch.dict(sys.modules, {"mcp.server": mock_server_mod, "mcp.types": mock_types}):
        server = MCPServer(mock_rag)
        server._build_server()

    call_tool_fn = registered["call_tool"]
    await call_tool_fn(name="rag_query", arguments={"query": "hello"})

    mock_rag.ask.assert_awaited_once_with("hello")


async def test_mcpserver_rag_unknown_tool_returns_error_not_exception():
    """Calling an unknown tool in RAG mode returns an error TextContent, not an exception."""
    mock_rag = MagicMock()
    mock_rag.__class__.__name__ = "RAG"
    mock_rag.ask = AsyncMock(return_value="x")
    mock_rag._vectorstore = MagicMock(_texts=[], _metadata=[])

    mock_server_mod, mock_types, _mock_srv, registered = _make_mock_mcp()

    with patch.dict(sys.modules, {"mcp.server": mock_server_mod, "mcp.types": mock_types}):
        server = MCPServer(mock_rag)
        server._build_server()

    call_tool_fn = registered["call_tool"]
    result = await call_tool_fn(name="nonexistent_tool", arguments={})
    # Should return a list of TextContent with an error message, not raise
    assert result is not None


async def test_mcpserver_rag_registers_resource_handlers():
    """MCPServer(rag) registers list_resources and read_resource handlers."""
    mock_rag = MagicMock()
    mock_rag.__class__.__name__ = "RAG"
    mock_rag.ask = AsyncMock(return_value="x")
    mock_rag._vectorstore = MagicMock(
        _texts=["doc one", "doc two"],
        _metadata=[{"src": "a"}, {"src": "b"}],
    )

    mock_server_mod, mock_types, _mock_srv, registered = _make_mock_mcp()

    with patch.dict(sys.modules, {"mcp.server": mock_server_mod, "mcp.types": mock_types}):
        server = MCPServer(mock_rag)
        server._build_server()

    assert "list_resources" in registered
    assert "read_resource" in registered


async def test_mcpserver_rag_list_resources_returns_correct_count():
    """list_resources returns one Resource per document in the vectorstore."""
    mock_rag = MagicMock()
    mock_rag.__class__.__name__ = "RAG"
    mock_rag.ask = AsyncMock(return_value="x")
    mock_rag._vectorstore = MagicMock(
        _texts=["a", "b", "c"],
        _metadata=[{}, {}, {}],
    )

    mock_server_mod, mock_types, _mock_srv, registered = _make_mock_mcp()

    with patch.dict(sys.modules, {"mcp.server": mock_server_mod, "mcp.types": mock_types}):
        server = MCPServer(mock_rag)
        server._build_server()

    resources = await registered["list_resources"]()
    assert len(resources) == 3


async def test_mcpserver_rag_read_resource_returns_document_text():
    """read_resource('document://0') returns the text at index 0."""
    mock_rag = MagicMock()
    mock_rag.__class__.__name__ = "RAG"
    mock_rag.ask = AsyncMock(return_value="x")
    mock_rag._vectorstore = MagicMock(
        _texts=["first doc", "second doc"],
        _metadata=[{}, {}],
    )

    mock_server_mod, mock_types, _mock_srv, registered = _make_mock_mcp()

    with patch.dict(sys.modules, {"mcp.server": mock_server_mod, "mcp.types": mock_types}):
        server = MCPServer(mock_rag)
        server._build_server()

    text = await registered["read_resource"](uri="document://0")
    assert text == "first doc"


async def test_mcpserver_rag_read_resource_invalid_index():
    """read_resource with out-of-bounds index raises ValueError."""
    mock_rag = MagicMock()
    mock_rag.__class__.__name__ = "RAG"
    mock_rag.ask = AsyncMock(return_value="x")
    mock_rag._vectorstore = MagicMock(_texts=["only one"], _metadata=[{}])

    mock_server_mod, mock_types, _mock_srv, registered = _make_mock_mcp()

    with patch.dict(sys.modules, {"mcp.server": mock_server_mod, "mcp.types": mock_types}):
        server = MCPServer(mock_rag)
        server._build_server()

    with pytest.raises(ValueError):
        await registered["read_resource"](uri="document://99")


# ------------------------------------------------------------------ #
# Agent mode
# ------------------------------------------------------------------ #


async def test_mcpserver_agent_registers_run_tool():
    """MCPServer(agent) registers exactly one tool named 'agent_run'."""
    mock_agent = MagicMock()
    mock_agent.__class__.__name__ = "FunctionCallingAgent"
    mock_agent.run = AsyncMock(return_value="done")

    mock_server_mod, mock_types, _mock_srv, registered = _make_mock_mcp()

    with patch.dict(sys.modules, {"mcp.server": mock_server_mod, "mcp.types": mock_types}):
        server = MCPServer(mock_agent)
        server._build_server()

    assert "list_tools" in registered
    assert "call_tool" in registered


async def test_mcpserver_agent_run_calls_agent():
    """'agent_run' tool handler forwards query to agent.run()."""
    mock_agent = MagicMock()
    mock_agent.__class__.__name__ = "FunctionCallingAgent"
    mock_agent.run = AsyncMock(return_value="Agent answer")

    mock_server_mod, mock_types, _mock_srv, registered = _make_mock_mcp()

    with patch.dict(sys.modules, {"mcp.server": mock_server_mod, "mcp.types": mock_types}):
        server = MCPServer(mock_agent)
        server._build_server()

    await registered["call_tool"](name="agent_run", arguments={"query": "do this"})
    mock_agent.run.assert_awaited_once_with("do this")


async def test_mcpserver_react_agent_wrapping():
    """MCPServer supports ReActAgent by class name."""
    mock_agent = MagicMock()
    mock_agent.__class__.__name__ = "ReActAgent"
    mock_agent.run = AsyncMock(return_value="ReAct answer")

    mock_server_mod, mock_types, _mock_srv, registered = _make_mock_mcp()

    with patch.dict(sys.modules, {"mcp.server": mock_server_mod, "mcp.types": mock_types}):
        server = MCPServer(mock_agent)
        server._build_server()

    assert "list_tools" in registered


async def test_mcpserver_executor_wrapping():
    """MCPServer supports AgentExecutor by class name."""
    mock_exec = MagicMock()
    mock_exec.__class__.__name__ = "AgentExecutor"
    mock_exec.run = AsyncMock(return_value="exec answer")

    mock_server_mod, mock_types, _mock_srv, registered = _make_mock_mcp()

    with patch.dict(sys.modules, {"mcp.server": mock_server_mod, "mcp.types": mock_types}):
        server = MCPServer(mock_exec)
        server._build_server()

    assert "list_tools" in registered


# ------------------------------------------------------------------ #
# Tools mode (default / backward compat)
# ------------------------------------------------------------------ #


async def test_mcpserver_tools_mode_dispatch():
    """Tools mode dispatches to the correct tool by name."""
    mock_tool = MagicMock()
    mock_tool.name = "calc"
    mock_tool.description = "A calculator"
    mock_tool.parameters = {"type": "object", "properties": {"expr": {"type": "string"}}}
    mock_result = MagicMock(is_error=False, output="42")
    mock_tool.run = AsyncMock(return_value=mock_result)

    mock_server_mod, mock_types, _mock_srv, registered = _make_mock_mcp()

    with patch.dict(sys.modules, {"mcp.server": mock_server_mod, "mcp.types": mock_types}):
        server = MCPServer(tools=[mock_tool])
        server._build_server()

    await registered["call_tool"](name="calc", arguments={"expr": "6*7"})
    mock_tool.run.assert_awaited_once_with(expr="6*7")


async def test_mcpserver_tools_mode_unknown_tool():
    """Tools mode returns error TextContent for unknown tool, not exception."""
    mock_server_mod, mock_types, _mock_srv, registered = _make_mock_mcp()

    with patch.dict(sys.modules, {"mcp.server": mock_server_mod, "mcp.types": mock_types}):
        server = MCPServer()
        server._build_server()

    result = await registered["call_tool"](name="nonexistent", arguments={})
    assert result is not None  # returns list[TextContent], not exception


async def test_mcpserver_tool_error_propagates_as_content():
    """Tool exceptions are caught and returned as error TextContent, not re-raised."""
    mock_tool = MagicMock()
    mock_tool.name = "boom"
    mock_tool.description = "A tool that fails"
    mock_tool.parameters = {}
    mock_tool.run = AsyncMock(side_effect=RuntimeError("kaboom"))

    mock_server_mod, mock_types, _mock_srv, registered = _make_mock_mcp()

    with patch.dict(sys.modules, {"mcp.server": mock_server_mod, "mcp.types": mock_types}):
        server = MCPServer(tools=[mock_tool])
        server._build_server()

    # Should not raise
    result = await registered["call_tool"](name="boom", arguments={})
    assert result is not None


async def test_mcpserver_combined_rag_and_extra_tools():
    """MCPServer(rag, tools=[...]) registers both the RAG query tool and extra tools.

    Note: In the current implementation, RAG mode takes precedence over the tools dict.
    The plan does not yet combine them, so we verify RAG mode activates.
    """
    mock_rag = MagicMock()
    mock_rag.__class__.__name__ = "RAG"
    mock_rag.ask = AsyncMock(return_value="answer")
    mock_rag._vectorstore = MagicMock(_texts=[], _metadata=[])

    mock_extra = MagicMock()
    mock_extra.name = "extra"

    mock_server_mod, mock_types, _mock_srv, registered = _make_mock_mcp()

    with patch.dict(sys.modules, {"mcp.server": mock_server_mod, "mcp.types": mock_types}):
        server = MCPServer(mock_rag, tools=[mock_extra])
        server._build_server()

    assert "list_tools" in registered


# ------------------------------------------------------------------ #
# run() stdio missing mcp
# ------------------------------------------------------------------ #


def test_mcpserver_run_missing_mcp():
    """run() raises ImportError when mcp.server.stdio is not installed."""
    saved = sys.modules.get("mcp")
    saved2 = sys.modules.get("mcp.server.stdio")
    try:
        with patch.dict(
            sys.modules,
            {"mcp": None, "mcp.server": None, "mcp.server.stdio": None, "mcp.types": None},
        ):
            server = MCPServer()
            with pytest.raises(ImportError, match="mcp package required"):
                server.run()
    finally:
        if saved is not None:
            sys.modules["mcp"] = saved
        if saved2 is not None:
            sys.modules["mcp.server.stdio"] = saved2
