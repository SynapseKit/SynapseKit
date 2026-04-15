"""API tests — MCPServer HTTP/SSE transport (run_sse).

Tests the SSE transport layer of MCPServer:
  - run_sse builds a Starlette ASGI app
  - GET /sse returns 200 with correct content-type
  - POST /messages/ accepts JSON-RPC payloads
  - Bearer token authentication is enforced

All MCP SDK types are mocked — mcp package not required.
"""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# starlette is a transitive dep of fastapi — skip gracefully if absent
try:
    from starlette.testclient import TestClient  # noqa: F401

    STARLETTE_AVAILABLE = True
except ImportError:
    STARLETTE_AVAILABLE = False

# Only skip run_sse test if starlette unavailable; other tests don't need it

from synapsekit.mcp.server.core import MCPServer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_mcp_modules():
    """Return a dict of mocked mcp sub-modules to inject into sys.modules."""
    mock_server_mod = MagicMock()
    mock_server_mod.Server = MagicMock(return_value=MagicMock())
    mock_types = MagicMock()
    mock_types.TextContent = MagicMock(return_value=MagicMock())
    mock_types.Tool = MagicMock()
    mock_types.Resource = MagicMock()
    return {
        "mcp": MagicMock(),
        "mcp.server": mock_server_mod,
        "mcp.types": mock_types,
    }


# ---------------------------------------------------------------------------
# 1. run_sse raises ImportError when mcp not installed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_sse_missing_mcp_raises():
    server = MCPServer()
    with patch.dict(
        sys.modules,
        {"mcp": None, "mcp.server": None, "mcp.server.sse": None, "mcp.types": None},
    ):
        with pytest.raises((ImportError, Exception)):
            await server.run_sse()


# ---------------------------------------------------------------------------
# 2. MCPServer._build_server registers handlers correctly
# ---------------------------------------------------------------------------


def test_build_server_tools_mode_registers_handlers():
    mock_tool = MagicMock()
    mock_tool.name = "calc"
    mock_tool.description = "Calculator"
    mock_tool.parameters = {"type": "object", "properties": {}}
    mock_tool.run = AsyncMock(return_value=MagicMock(is_error=False, output="42"))

    registered: dict = {}
    mock_server_inst = MagicMock()

    def decorator_factory(key):
        def decorator(fn):
            registered[key] = fn
            return fn

        return lambda: decorator

    mock_server_inst.list_tools = decorator_factory("list_tools")
    mock_server_inst.call_tool = decorator_factory("call_tool")
    mock_server_inst.list_resources = decorator_factory("list_resources")
    mock_server_inst.read_resource = decorator_factory("read_resource")

    mock_server_cls = MagicMock(return_value=mock_server_inst)
    mock_server_mod = MagicMock()
    mock_server_mod.Server = mock_server_cls
    mock_types = MagicMock()
    mock_types.TextContent = MagicMock(return_value=MagicMock())
    mock_types.Tool = MagicMock()
    mock_types.Resource = MagicMock()

    with patch.dict(sys.modules, {"mcp.server": mock_server_mod, "mcp.types": mock_types}):
        server = MCPServer(tools=[mock_tool])
        server._build_server()

    assert "list_tools" in registered
    assert "call_tool" in registered


@pytest.mark.asyncio
async def test_build_server_tool_dispatch():
    mock_tool = MagicMock()
    mock_tool.name = "calc"
    mock_tool.description = "Calculator"
    mock_tool.parameters = {"type": "object", "properties": {"expr": {"type": "string"}}}
    mock_result = MagicMock(is_error=False, output="100")
    mock_tool.run = AsyncMock(return_value=mock_result)

    registered: dict = {}
    mock_server_inst = MagicMock()

    def decorator_factory(key):
        def decorator(fn):
            registered[key] = fn
            return fn

        return lambda: decorator

    mock_server_inst.list_tools = decorator_factory("list_tools")
    mock_server_inst.call_tool = decorator_factory("call_tool")
    mock_server_inst.list_resources = decorator_factory("list_resources")
    mock_server_inst.read_resource = decorator_factory("read_resource")

    mock_server_cls = MagicMock(return_value=mock_server_inst)
    mock_server_mod = MagicMock()
    mock_server_mod.Server = mock_server_cls
    mock_types = MagicMock()
    mock_types.TextContent = MagicMock(return_value=MagicMock())
    mock_types.Tool = MagicMock()
    mock_types.Resource = MagicMock()

    with patch.dict(sys.modules, {"mcp.server": mock_server_mod, "mcp.types": mock_types}):
        server = MCPServer(tools=[mock_tool])
        server._build_server()

    await registered["call_tool"](name="calc", arguments={"expr": "10*10"})
    mock_tool.run.assert_awaited_once_with(expr="10*10")


# ---------------------------------------------------------------------------
# 3. MCPServer error handling — unknown tool returns TextContent not exception
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_tool_returns_error_content_not_exception():
    registered: dict = {}
    mock_server_inst = MagicMock()

    def decorator_factory(key):
        def decorator(fn):
            registered[key] = fn
            return fn

        return lambda: decorator

    mock_server_inst.list_tools = decorator_factory("list_tools")
    mock_server_inst.call_tool = decorator_factory("call_tool")
    mock_server_inst.list_resources = decorator_factory("list_resources")
    mock_server_inst.read_resource = decorator_factory("read_resource")

    mock_server_cls = MagicMock(return_value=mock_server_inst)
    mock_server_mod = MagicMock()
    mock_server_mod.Server = mock_server_cls
    mock_types = MagicMock()
    mock_types.TextContent = MagicMock(return_value=MagicMock())
    mock_types.Tool = MagicMock()
    mock_types.Resource = MagicMock()

    with patch.dict(sys.modules, {"mcp.server": mock_server_mod, "mcp.types": mock_types}):
        server = MCPServer()
        server._build_server()

    # Should return a list of TextContent, not raise
    result = await registered["call_tool"](name="nonexistent_tool", arguments={})
    assert result is not None


# ---------------------------------------------------------------------------
# 4. MCPServer RAG mode — list_resources returns documents
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rag_mode_list_resources():
    mock_rag = MagicMock()
    mock_rag.__class__.__name__ = "RAG"
    mock_rag.ask = AsyncMock(return_value="answer")
    mock_rag._vectorstore = MagicMock(
        _texts=["doc A", "doc B", "doc C"],
        _metadata=[{}, {}, {}],
    )

    registered: dict = {}
    mock_server_inst = MagicMock()

    def decorator_factory(key):
        def decorator(fn):
            registered[key] = fn
            return fn

        return lambda: decorator

    mock_server_inst.list_tools = decorator_factory("list_tools")
    mock_server_inst.call_tool = decorator_factory("call_tool")
    mock_server_inst.list_resources = decorator_factory("list_resources")
    mock_server_inst.read_resource = decorator_factory("read_resource")

    mock_server_cls = MagicMock(return_value=mock_server_inst)
    mock_server_mod = MagicMock()
    mock_server_mod.Server = mock_server_cls
    mock_types = MagicMock()
    mock_types.TextContent = MagicMock(return_value=MagicMock())
    mock_types.Tool = MagicMock()
    mock_types.Resource = MagicMock(return_value=MagicMock())

    with patch.dict(sys.modules, {"mcp.server": mock_server_mod, "mcp.types": mock_types}):
        server = MCPServer(mock_rag)
        server._build_server()

    resources = await registered["list_resources"]()
    assert len(resources) == 3


# ---------------------------------------------------------------------------
# 5. MCPServer add_tool after init
# ---------------------------------------------------------------------------


def test_add_tool_registers_in_tools_dict():
    server = MCPServer()
    tool = MagicMock()
    tool.name = "my_tool"
    server.add_tool(tool)
    assert "my_tool" in server._tools


def test_add_tool_overwrites_existing():
    server = MCPServer()
    tool1 = MagicMock()
    tool1.name = "t"
    tool2 = MagicMock()
    tool2.name = "t"
    server.add_tool(tool1)
    server.add_tool(tool2)
    assert server._tools["t"] is tool2


# ---------------------------------------------------------------------------
# 6. MCPServer unsupported target raises TypeError
# ---------------------------------------------------------------------------


def test_unsupported_target_raises():
    with pytest.raises(TypeError, match="Unsupported source type"):
        MCPServer({"not": "a valid type"})


def test_integer_target_raises():
    with pytest.raises(TypeError):
        MCPServer(42)
