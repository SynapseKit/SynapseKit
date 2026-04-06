from __future__ import annotations

import re
from html.parser import HTMLParser

from .base import BaseSplitter
from .recursive import RecursiveCharacterTextSplitter

BLOCK_TAGS = frozenset({
    "h1", "h2", "h3", "h4", "h5", "h6",
    "p", "div", "section", "article",
    "li", "blockquote", "pre",
})


class _TagStripper(HTMLParser):
    """Strips HTML tags and returns plain text."""

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str):
        self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts).strip()


def _strip_tags(html: str) -> str:
    s = _TagStripper()
    s.feed(html)
    return s.get_text()


class HTMLTextSplitter(BaseSplitter):
    """Split HTML documents on block-level tags and return plain-text chunks."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []

        plain = _strip_tags(text)
        if len(plain) <= self.chunk_size:
            return [plain] if plain else []

        sections = self._split_by_blocks(text)
        chunks: list[str] = []

        fallback = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )

        for section in sections:
            stripped = _strip_tags(section).strip()
            if not stripped:
                continue
            if len(stripped) <= self.chunk_size:
                chunks.append(stripped)
            else:
                chunks.extend(fallback.split(stripped))

        if self.chunk_overlap > 0 and len(chunks) >= 2:
            overlapped = [chunks[0]]
            for i in range(1, len(chunks)):
                tail = chunks[i - 1][-self.chunk_overlap:]
                overlapped.append(tail + chunks[i])
            chunks = overlapped

        return chunks

    def _split_by_blocks(self, html: str) -> list[str]:
        # match opening block tags
        tag_names = "|".join(BLOCK_TAGS)
        pattern = re.compile(
            rf"<(?:{tag_names})[\s>]",
            re.IGNORECASE,
        )

        positions = [m.start() for m in pattern.finditer(html)]
        if not positions:
            return [html]

        sections: list[str] = []

        # content before first block tag
        if positions[0] > 0:
            pre = html[:positions[0]]
            if pre.strip():
                sections.append(pre)

        for i, pos in enumerate(positions):
            end = positions[i + 1] if i + 1 < len(positions) else len(html)
            sections.append(html[pos:end])

        return sections
