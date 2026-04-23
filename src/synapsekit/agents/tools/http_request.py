from __future__ import annotations

from typing import Any

from ..base import BaseTool, ToolResult


class HTTPRequestTool(BaseTool):
    """Make HTTP requests (GET, POST, PUT, DELETE, PATCH).

    A single ``aiohttp.ClientSession`` is created lazily on the first request
    and reused for all subsequent calls on the same tool instance.  This
    preserves TCP connection pooling and avoids the overhead of a new TLS
    handshake on every call.

    Call ``await tool.aclose()`` (or use as an async context manager) to
    release the underlying connection pool when the tool is no longer needed.
    """

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
        self._session: Any | None = None  # aiohttp.ClientSession, created lazily

    async def _get_session(self) -> Any:
        """Return the shared session, creating it on first use."""
        try:
            import aiohttp
        except ImportError:
            raise ImportError("aiohttp required for HTTPRequestTool: pip install aiohttp") from None

        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def aclose(self) -> None:
        """Close the underlying connection pool."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> HTTPRequestTool:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

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

        method = method.upper()
        req_headers = headers or {}

        # _get_session raises ImportError if aiohttp is missing — let it propagate
        session = await self._get_session()

        try:
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
