from __future__ import annotations

from .base import Document


class WebLoader:
    """Fetch a URL and return its text content as a Document."""

    def __init__(self, url: str) -> None:
        self._url = url

    def _parse(self, html: str) -> str:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("beautifulsoup4 required: pip install synapsekit[web]") from None
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator="\n", strip=True)

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
