"""Core MCPServer implementation."""

from __future__ import annotations

import json
from typing import Any

from ...agents.base import BaseTool


def _make_server(server_cls: Any, name: str, handlers: dict[str, Any]) -> Any:
    if hasattr(server_cls, "list_tools"):
        # mcp 1.x: register the handlers with decorators.
        server = server_cls(name)
        for key, fn in handlers.items():
            getattr(server, key)()(fn)
        return server

    # mcp 2.x: the handlers go to the constructor and return result models.
    from mcp.types import (
        CallToolResult,
        ListResourcesResult,
        ListToolsResult,
        ReadResourceResult,
        TextResourceContents,
    )

    async def on_list_tools(ctx: Any, params: Any) -> Any:
        return ListToolsResult(tools=await handlers["list_tools"]())

    async def on_call_tool(ctx: Any, params: Any) -> Any:
        content = await handlers["call_tool"](params.name, params.arguments)
        return CallToolResult(content=content)

    kwargs: dict[str, Any] = {"on_list_tools": on_list_tools, "on_call_tool": on_call_tool}
    if "list_resources" in handlers:

        async def on_list_resources(ctx: Any, params: Any) -> Any:
            return ListResourcesResult(resources=await handlers["list_resources"]())

        async def on_read_resource(ctx: Any, params: Any) -> Any:
            text = await handlers["read_resource"](params.uri)
            contents = [TextResourceContents(uri=params.uri, mime_type="text/plain", text=text)]
            return ReadResourceResult(contents=contents)

        kwargs["on_list_resources"] = on_list_resources
        kwargs["on_read_resource"] = on_read_resource
    return server_cls(name, **kwargs)


class MCPServer:
    """Expose SynapseKit tools, RAG, or Agents as an MCP server.

    Requires: pip install mcp

    Usage::

        from synapsekit import MCPServer, CalculatorTool, DateTimeTool

        # Tools mode
        server = MCPServer(tools=[CalculatorTool(), DateTimeTool()])

        # RAG mode
        server = MCPServer(rag)

        # Agent mode
        server = MCPServer(agent)

        server.run()  # starts stdio server
    """

    #: Class names accepted as a source target (RAG / agent types).
    _SUPPORTED_TARGET_NAMES: frozenset[str] = frozenset(
        {
            "RAG",
            "RAGPipeline",
            "FunctionCallingAgent",
            "ReActAgent",
            "AgentExecutor",
            "KnowledgeMesh",
        }
    )

    def __init__(
        self,
        target: Any = None,
        *,
        name: str | None = None,
        tools: list[BaseTool] | None = None,
        version: str = "1.0.0",
        **kwargs: Any,
    ) -> None:
        # Backwards compatibility: if target is a string, it was passed as `name`
        if isinstance(target, str):
            name = target
            target = None

        # Validate target type — only None or known SynapseKit classes are allowed.
        if target is not None:
            cls_name = type(target).__name__
            if cls_name not in self._SUPPORTED_TARGET_NAMES:
                raise TypeError(
                    f"Unsupported source type: {type(target)!r}. "
                    f"MCPServer accepts RAG, FunctionCallingAgent, ReActAgent, "
                    f"or AgentExecutor as a positional argument. "
                    f"Pass individual tools via tools=[...]."
                )

        self._target = target
        self._name = name or "synapsekit"
        self._tools: dict[str, BaseTool] = {t.name: t for t in (tools or [])}
        self._version = version
        self._server: Any = None

    def add_tool(self, tool: BaseTool) -> None:
        """Add a tool to the server (tools mode)."""
        self._tools[tool.name] = tool

    def _build_server(self) -> Any:
        try:
            from mcp.server import Server
            from mcp.types import Resource, TextContent, Tool
        except ImportError:
            raise ImportError("mcp package required: pip install mcp") from None

        target = self._target
        resource_handlers: dict[str, Any] = {}

        # ------------------------------------------------------------------ #
        # RAG Mode
        # ------------------------------------------------------------------ #
        if target is not None and type(target).__name__ == "KnowledgeMesh":
            tools_map = {tool.name: tool for tool in target.as_mcp_tools()}

            async def list_tools() -> list[Tool]:
                return [
                    Tool(
                        name=tool.name,
                        description=tool.description,
                        inputSchema=getattr(tool, "parameters", {}),
                    )
                    for tool in tools_map.values()
                ]

            async def call_tool(
                name: str, arguments: dict[str, Any] | None = None
            ) -> list[TextContent]:
                tool = tools_map.get(name)
                if tool is None:
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]

                try:
                    res = await tool.run(**(arguments or {}))
                    if res.is_error:
                        return [TextContent(type="text", text=f"Error: {res.error}")]
                    return [TextContent(type="text", text=res.output)]
                except Exception as e:
                    return [TextContent(type="text", text=f"Tool error: {e}")]

        elif target is not None and type(target).__name__ in ("RAG", "RAGPipeline"):

            async def list_tools() -> list[Tool]:
                return [
                    Tool(
                        name="rag_query",
                        description="Query the RAG knowledge base.",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "The search query"}
                            },
                            "required": ["query"],
                        },
                    )
                ]

            async def call_tool(
                name: str, arguments: dict[str, Any] | None = None
            ) -> list[TextContent]:
                if name == "rag_query":
                    query = (arguments or {}).get("query", "")
                    try:
                        # RAG has .ask() or .arun()
                        if hasattr(target, "ask"):
                            ans = await target.ask(query)
                        elif hasattr(target, "run"):
                            # some generic run
                            ans = await target.run(query)
                        else:
                            ans = "Target does not support ask() or run()"
                        return [TextContent(type="text", text=str(ans))]
                    except Exception as e:
                        return [TextContent(type="text", text=f"Error querying RAG: {e}")]
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

            async def list_resources() -> list[Resource]:
                resources = []
                vs = getattr(target, "_vectorstore", None)
                if vs and hasattr(vs, "_texts") and hasattr(vs, "_metadata"):
                    for i, meta in enumerate(vs._metadata):
                        uri = f"document://{i}"
                        resources.append(
                            Resource(
                                uri=uri,
                                name=f"Document {i}",
                                mimeType="text/plain",
                                description=json.dumps(meta) if meta else "RAG Document",
                            )
                        )
                return resources

            async def read_resource(uri: str | Any) -> str | bytes:
                uri_str = str(uri)
                vs = getattr(target, "_vectorstore", None)
                if vs and hasattr(vs, "_texts") and uri_str.startswith("document://"):
                    try:
                        idx = int(uri_str.split("://")[1])
                        return str(vs._texts[idx])
                    except (ValueError, IndexError):
                        raise ValueError(f"Resource not found: {uri_str}") from None
                raise ValueError(f"Resource not found: {uri_str}")

            resource_handlers = {"list_resources": list_resources, "read_resource": read_resource}

        # ------------------------------------------------------------------ #
        # Agent Mode
        # ------------------------------------------------------------------ #
        elif target is not None and type(target).__name__ in (
            "FunctionCallingAgent",
            "AgentExecutor",
            "ReActAgent",
        ):

            async def list_tools() -> list[Tool]:
                return [
                    Tool(
                        name="agent_run",
                        description="Send a task or query to the agent.",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "The task or question"}
                            },
                            "required": ["query"],
                        },
                    )
                ]

            async def call_tool(
                name: str, arguments: dict[str, Any] | None = None
            ) -> list[TextContent]:
                if name == "agent_run":
                    query = (arguments or {}).get("query", "")
                    try:
                        ans = await target.run(query)
                        return [TextContent(type="text", text=str(ans))]
                    except Exception as e:
                        return [TextContent(type="text", text=f"Error running agent: {e}")]
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

        # ------------------------------------------------------------------ #
        # Tools Mode (default)
        # ------------------------------------------------------------------ #
        else:
            tools_map = self._tools

            async def list_tools() -> list[Tool]:
                result = []
                for t in tools_map.values():
                    result.append(
                        Tool(
                            name=t.name,
                            description=t.description,
                            inputSchema=t.parameters if hasattr(t, "parameters") else {},
                        )
                    )
                return result

            async def call_tool(
                name: str, arguments: dict[str, Any] | None = None
            ) -> list[TextContent]:
                tool = tools_map.get(name)
                if not tool:
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]

                try:
                    res = await tool.run(**(arguments or {}))
                    if res.is_error:
                        return [TextContent(type="text", text=f"Error: {res.error}")]
                    return [TextContent(type="text", text=res.output)]
                except Exception as e:
                    return [TextContent(type="text", text=f"Tool error: {e}")]

        handlers = {"list_tools": list_tools, "call_tool": call_tool, **resource_handlers}
        server = _make_server(Server, self._name, handlers)
        self._server = server
        return server

    def run(self) -> None:
        """Run as stdio MCP server (blocking)."""
        try:
            from mcp.server.stdio import stdio_server
        except ImportError:
            raise ImportError("mcp package required: pip install mcp") from None

        import asyncio

        server = self._build_server()

        async def _run() -> None:
            async with stdio_server() as (read, write):
                await server.run(read, write, server.create_initialization_options())

        asyncio.run(_run())

    async def run_sse(
        self, host: str = "0.0.0.0", port: int = 8000, api_key: str | None = None
    ) -> None:
        """Run as SSE MCP server (async, blocking until stopped).

        Requires: pip install synapsekit[mcp,serve]

        Args:
            host: Bind address (default ``0.0.0.0``).
            port: TCP port (default ``8000``).
            api_key: Optional Bearer token for API key auth.
        """
        try:
            from mcp.server.sse import SseServerTransport
        except ImportError:
            raise ImportError("mcp package required: pip install mcp") from None

        try:
            import uvicorn
            from starlette.applications import Starlette
            from starlette.middleware.base import BaseHTTPMiddleware
            from starlette.responses import JSONResponse
            from starlette.routing import Route
        except ImportError:
            raise ImportError(
                "starlette and uvicorn are required for SSE transport: "
                "pip install synapsekit[serve]"
            ) from None

        server = self._build_server()
        sse = SseServerTransport("/messages/")

        async def handle_sse(request: Any) -> None:
            async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
                await server.run(streams[0], streams[1], server.create_initialization_options())

        async def messages(request: Any) -> None:
            await sse.handle_post_message(request.scope, request.receive, request._send)

        class AuthMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Any, call_next: Any) -> Any:
                if api_key:
                    auth = request.headers.get("Authorization")
                    if not auth or auth != f"Bearer {api_key}":
                        return JSONResponse({"error": "Unauthorized"}, status_code=401)
                return await call_next(request)

        app = Starlette(
            routes=[
                Route("/sse", endpoint=handle_sse),
                Route("/messages/", endpoint=messages, methods=["POST"]),
            ]
        )
        if api_key:
            app.add_middleware(AuthMiddleware)

        config = uvicorn.Config(app, host=host, port=port, log_level="info")
        uvicorn_server = uvicorn.Server(config)
        await uvicorn_server.serve()
