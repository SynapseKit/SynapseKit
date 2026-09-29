from __future__ import annotations

import os
import sys
import textwrap

import pytest

pytest.importorskip("mcp")

from synapsekit import CalculatorTool
from synapsekit.mcp.client import MCPClient

_TOOLS_SERVER = (
    "from synapsekit import CalculatorTool, MCPServer; MCPServer(tools=[CalculatorTool()]).run()"
)

_RAG_SERVER = textwrap.dedent(
    """
    from synapsekit.mcp import MCPServer

    class _Store:
        _texts = ["alpha", "beta"]
        _metadata = [{}, {}]

    class RAG:
        _vectorstore = _Store()

    MCPServer(RAG()).run()
    """
)


def _env() -> dict[str, str]:
    # The server needs the same synapsekit and mcp.
    return {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}


async def test_client_and_server_round_trip_over_stdio():
    client = MCPClient()
    try:
        tools = await client.connect_stdio(sys.executable, ["-c", _TOOLS_SERVER], env=_env())
        assert [t.name for t in tools] == ["calculator"]
        assert tools[0].parameters == CalculatorTool().parameters
        result = await tools[0].run(expression="2 + 3")
        assert not result.is_error, result.error
        assert result.output == "5"
    finally:
        await client.close()


async def test_rag_server_resources_over_stdio():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=sys.executable, args=["-c", _RAG_SERVER], env=_env())
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        listed = await session.list_resources()
        assert [str(r.uri) for r in listed.resources] == ["document://0", "document://1"]
        result = await session.read_resource(listed.resources[1].uri)
        assert [c.text for c in result.contents] == ["beta"]
