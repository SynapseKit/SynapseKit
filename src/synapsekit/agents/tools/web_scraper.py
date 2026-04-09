from __future__ import annotations

import ipaddress
import socket
from typing import Any
from urllib.parse import urlparse

from ..base import BaseTool, ToolResult

_PRIVATE_NETS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
]
_MAX_CSS_SELECTOR_LEN = 200


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"URL scheme {parsed.scheme!r} is not allowed; use http or https.")
    host = parsed.hostname or ""
    if not host:
        raise ValueError("URL has no hostname.")
    try:
        addr = ipaddress.ip_address(socket.gethostbyname(host))
    except (socket.gaierror, ValueError):
        return
    if any(addr in net for net in _PRIVATE_NETS):
        raise ValueError(f"Requests to private/internal addresses are not allowed: {host!r}")


class WebScraperTool(BaseTool):
    """Fetch a URL and extract clean text content from the webpage."""

    name = "web_scraper"
    description = (
        "Fetch a webpage URL and return clean text content. "
        "Input: a URL string, e.g. 'https://example.com/article'. "
        "Optional: css_selector to extract specific content. "
        "Removes scripts, styles, nav, header, footer elements."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL of the webpage to scrape",
            },
            "css_selector": {
                "type": "string",
                "description": "Optional CSS selector to extract specific content (e.g. 'article', '.content')",
            },
        },
        "required": ["url"],
    }

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def _clean_html(self, html: str, css_selector: str | None = None) -> str:
        """Parse HTML and extract clean text."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("beautifulsoup4 required: pip install synapsekit[web]") from None

        soup = BeautifulSoup(html, "html.parser")

        for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
            element.decompose()

        if css_selector:
            if len(css_selector) > _MAX_CSS_SELECTOR_LEN:
                raise ValueError(
                    f"CSS selector exceeds maximum length of {_MAX_CSS_SELECTOR_LEN} characters."
                )
            elements = soup.select(css_selector)
            if not elements:
                return ""
            content = "\n\n".join(elem.get_text(separator="\n", strip=True) for elem in elements)
        else:
            content = soup.get_text(separator="\n", strip=True)

        return content

    async def run(
        self, url: str = "", css_selector: str | None = None, **kwargs: Any
    ) -> ToolResult:
        """Fetch URL and return clean text."""
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required: pip install synapsekit[web]") from None

        target_url = url or kwargs.get("input", "")
        if not target_url:
            return ToolResult(output="", error="No URL provided.")

        try:
            _validate_url(target_url)
        except ValueError as e:
            return ToolResult(output="", error=str(e))

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(target_url, timeout=self.timeout)
                response.raise_for_status()
                text = self._clean_html(response.text, css_selector)
                return ToolResult(output=text)
        except httpx.HTTPError as e:
            return ToolResult(output="", error=f"HTTP error: {e}")
        except Exception as e:
            return ToolResult(output="", error=f"Scraping failed: {e}")
