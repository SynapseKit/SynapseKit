from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

from .base import Document

_PRIVATE_NETS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
]


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


class WebLoader:
    """Fetch a URL and return its text content as a Document."""

    def __init__(self, url: str) -> None:
        _validate_url(url)
        self._url = url

    def _parse(self, html: str) -> str:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("beautifulsoup4 required: pip install synapsekit[web]") from None
        soup = BeautifulSoup(html, "html.parser")
        return str(soup.get_text(separator="\n", strip=True))

    async def load(self) -> list[Document]:
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required: pip install synapsekit[web]") from None

        async with httpx.AsyncClient() as client:
            response = await client.get(self._url)
            response.raise_for_status()

        text = self._parse(response.text)
        return [Document(text=text, metadata={"source": self._url})]

    def load_sync(self) -> list[Document]:
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required: pip install synapsekit[web]") from None

        with httpx.Client() as client:
            response = client.get(self._url)
            response.raise_for_status()

        text = self._parse(response.text)
        return [Document(text=text, metadata={"source": self._url})]
