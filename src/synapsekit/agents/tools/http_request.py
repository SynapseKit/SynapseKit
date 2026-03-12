from __future__ import annotations

from typing import Any

from ..base import BaseTool, ToolResult


class HTTPRequestTool(BaseTool):
    """Make HTTP requests (GET, POST, PUT, DELETE, PATCH)."""

    name = "http_request"
    description = (
        "Make an HTTP request to a URL. "
        "Input: method (GET/POST/PUT/DELETE/PATCH), url, optional body and headers."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to request"},
            "method": {
                "type": "string",
                "description": "HTTP method (default: GET)",
                "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                "default": "GET",
            },
            "body": {
                "type": "string",
                "description": "Request body (for POST/PUT/PATCH)",
                "default": "",
            },
            "headers": {
                "type": "object",
                "description": "HTTP headers as key-value pairs",
                "default": {},
            },
        },
        "required": ["url"],
    }

    def __init__(self, max_response_length: int = 10000, timeout: int = 30) -> None:
        self._max_length = max_response_length
        self._timeout = timeout

    async def run(
        self,
        url: str = "",
        method: str = "GET",
        body: str = "",
        headers: dict | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        url = url or kwargs.get("input", "")
        if not url:
            return ToolResult(output="", error="No URL provided.")

        try:
            import aiohttp
        except ImportError:
            raise ImportError("aiohttp required for HTTPRequestTool: pip install aiohttp") from None

        method = method.upper()
        req_headers = headers or {}

        try:
            timeout = aiohttp.ClientTimeout(total=self._timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                req_kwargs: dict[str, Any] = {"headers": req_headers}
                if method in ("POST", "PUT", "PATCH") and body:
                    req_kwargs["data"] = body

                async with session.request(method, url, **req_kwargs) as resp:
                    status = resp.status
                    text = await resp.text()
                    if len(text) > self._max_length:
                        text = text[: self._max_length] + "\n... (truncated)"
                    return ToolResult(output=f"HTTP {status}\n{text}")
        except Exception as e:
            return ToolResult(output="", error=f"HTTP request failed: {e}")
