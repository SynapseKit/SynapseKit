from __future__ import annotations

from .base import Document


class HTMLLoader:
    """Load an HTML file, stripping tags to plain text."""

    def __init__(self, path: str, encoding: str = "utf-8") -> None:
        self._path = path
        self._encoding = encoding

    def load(self) -> list[Document]:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("beautifulsoup4 required: pip install synapsekit[html]") from None

        with open(self._path, encoding=self._encoding) as f:
            content = f.read()
        soup = BeautifulSoup(content, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        return [Document(text=text, metadata={"source": self._path})]
