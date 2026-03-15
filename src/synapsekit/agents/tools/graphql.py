from __future__ import annotations

import json
from typing import Any

from ..base import BaseTool, ToolResult


class GraphQLTool(BaseTool):
    """Execute GraphQL queries against an endpoint."""

    name = "graphql_query"
    description = (
        "Execute a GraphQL query against an endpoint. "
        "Input: url, query, optional variables (JSON string), optional headers (JSON string)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The GraphQL endpoint URL",
            },
            "query": {
                "type": "string",
                "description": "The GraphQL query string",
            },
            "variables": {
                "type": "string",
                "description": "JSON string of variables (optional)",
                "default": "",
            },
            "headers": {
                "type": "string",
                "description": "JSON string of HTTP headers (optional)",
                "default": "",
            },
        },
        "required": ["url", "query"],
    }

    def __init__(self, timeout: int = 30) -> None:
        self._timeout = timeout

    async def run(
        self,
        url: str = "",
        query: str = "",
        variables: str | None = None,
        headers: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        if not url:
            return ToolResult(output="", error="No URL provided.")
        if not query:
            return ToolResult(output="", error="No GraphQL query provided.")

        try:
            import aiohttp
        except ImportError:
            raise ImportError(
                "aiohttp required for GraphQLTool: pip install synapsekit[http]"
            ) from None

        # Parse optional JSON strings
        parsed_variables: dict | None = None
        if variables:
            try:
                parsed_variables = json.loads(variables)
            except json.JSONDecodeError:
                return ToolResult(output="", error="Invalid JSON in variables.")

        parsed_headers: dict = {"Content-Type": "application/json"}
        if headers:
            try:
                extra = json.loads(headers)
                parsed_headers.update(extra)
            except json.JSONDecodeError:
                return ToolResult(output="", error="Invalid JSON in headers.")

        payload: dict[str, Any] = {"query": query}
        if parsed_variables:
            payload["variables"] = parsed_variables

        try:
            timeout = aiohttp.ClientTimeout(total=self._timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=parsed_headers) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        return ToolResult(output="", error=f"HTTP {resp.status}: {text}")
                    data = await resp.json()
                    return ToolResult(output=json.dumps(data, indent=2))
        except Exception as e:
            return ToolResult(output="", error=f"GraphQL request failed: {e}")
