from __future__ import annotations

import asyncio
import os
from html.parser import HTMLParser

from .base import Document


class _TextExtractor(HTMLParser):
    """Minimal HTMLParser subclass that collects visible text."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_tags = {"script", "style"}
        self._skip = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._skip_tags:
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in self._skip_tags:
            self._skip = False

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._parts.append(data)

    def text(self) -> str:
        return (" ".join(self._parts).split() and " ".join(" ".join(self._parts).split())) or ""


# Uses HTMLParser for safe text extraction (avoids regex-based parsing issues)
def _html_to_text(html: bytes | str) -> str:
    # HTML parsing is best-effort; complex formatting may not be fully preserved
    parser = _TextExtractor()
    parser.feed(html if isinstance(html, str) else html.decode("utf-8", errors="ignore"))
    return " ".join(" ".join(parser._parts).split())


class EPUBLoader:
    """Load an EPUB file and extract text chapter-by-chapter.

    Requires: pip install synapsekit[epub]
    """

    def __init__(self, path: str) -> None:
        self._path = path

    def load(self) -> list[Document]:
        if not os.path.exists(self._path):
            raise FileNotFoundError(f"EPUB file not found: {self._path}")
        try:
            from ebooklib import epub
        except ImportError:
            raise ImportError("ebooklib required: pip install synapsekit[epub]") from None

        book = epub.read_epub(self._path)

        raw_title = book.get_metadata("DC", "title")
        raw_author = book.get_metadata("DC", "creator")
        title = raw_title[0][0] if raw_title and raw_title[0] else ""
        author = raw_author[0][0] if raw_author and raw_author[0] else ""

        docs = []
        for item in book.get_items():
            if item.get_type() != epub.ITEM_DOCUMENT:
                continue
            text = _html_to_text(item.get_content())
            if not text:
                continue
            docs.append(
                Document(
                    text=text,
                    metadata={
                        "source": self._path,
                        "title": title,
                        "author": author,
                        "chapter": item.get_name(),
                    },
                )
            )
        return docs

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)
