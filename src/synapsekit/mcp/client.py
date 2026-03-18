"""MCP Client -- connect SynapseKit agents to MCP tool servers."""

from __future__ import annotations

from typing import Any

from ..agents.base import BaseTool, ToolResult


class MCPToolAdapter(BaseTool):
    """Wraps an MCP tool as a SynapseKit BaseTool."""

    def __init__(self, mcp_tool: Any) -> None:
        self._mcp_tool = mcp_tool
        self.name = mcp_tool.name
        self.description = mcp_tool.description or f"MCP tool: {mcp_tool.name}"
        self.parameters = mcp_tool.inputSchema if hasattr(mcp_tool, "inputSchema") else {}

    async def run(self, **kwargs: Any) -> ToolResult:
        return ToolResult(output="", error="Tool not connected to an MCP session.")


class MCPClient:
    """Connect to an MCP server and use its tools as SynapseKit tools.

    Requires: pip install mcp

    Usage::

        client = MCPClient()
        tools = await client.connect_stdio("uvx", ["some-mcp-server"])
        # tools is a list of BaseTool instances

        # Use with an agent
        executor = AgentExecutor(AgentConfig(llm=llm, tools=tools))
    """

    def __init__(self) -> None:
        self._session: Any = None
        self._tools: list[MCPToolAdapter] = []

    async def connect_stdio(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> list[BaseTool]:
        """Connect to an MCP server via stdio transport."""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            raise ImportError("mcp package required: pip install mcp") from None

        params = StdioServerParameters(command=command, args=args or [], env=env)

        self._stdio_transport = stdio_client(params)
        streams = await self._stdio_transport.__aenter__()
        read_stream, write_stream = streams

        self._session = ClientSession(read_stream, write_stream)
        await self._session.__aenter__()
        await self._session.initialize()

        return await self._load_tools()

    async def connect_sse(self, url: str, headers: dict[str, str] | None = None) -> list[BaseTool]:
        """Connect to an MCP server via SSE transport."""
        try:
            from mcp import ClientSession
            from mcp.client.sse import sse_client
        except ImportError:
            raise ImportError("mcp package required: pip install mcp") from None

        self._sse_transport = sse_client(url, headers=headers or {})
        streams = await self._sse_transport.__aenter__()
        read_stream, write_stream = streams

        self._session = ClientSession(read_stream, write_stream)
        await self._session.__aenter__()
        await self._session.initialize()

        return await self._load_tools()

    async def _load_tools(self) -> list[BaseTool]:
        """Load tools from the connected MCP server."""
        if not self._session:
            raise RuntimeError("Not connected to an MCP server.")

        result = await self._session.list_tools()
        self._tools = []

        for mcp_tool in result.tools:
            adapter = MCPToolAdapter(mcp_tool)
            session = self._session

            async def _make_run(tool_name: str, _session: Any = session):
                async def _run(**kwargs: Any) -> ToolResult:
                    try:
                        call_result = await _session.call_tool(tool_name, arguments=kwargs)
                        output_parts = []
                        for content in call_result.content:
                            if hasattr(content, "text"):
                                output_parts.append(content.text)
                        output = (
                            "\n".join(output_parts) if output_parts else str(call_result.content)
                        )
                        if call_result.isError:
                            return ToolResult(output="", error=output)
                        return ToolResult(output=output)
                    except Exception as e:
                        return ToolResult(output="", error=f"MCP tool error: {e}")

                return _run

            adapter.run = await _make_run(mcp_tool.name)  # type: ignore[assignment]
            self._tools.append(adapter)

        return list(self._tools)

    async def close(self) -> None:
        """Close the MCP connection."""
        if self._session:
            await self._session.__aexit__(None, None, None)
        if hasattr(self, "_stdio_transport"):
            await self._stdio_transport.__aexit__(None, None, None)
        if hasattr(self, "_sse_transport"):
            await self._sse_transport.__aexit__(None, None, None)

    async def __aenter__(self) -> MCPClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
