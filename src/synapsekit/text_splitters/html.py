from __future__ import annotations

import html
import re

from .base import BaseSplitter


class HTMLTextSplitter(BaseSplitter):
    """Split HTML into plain-text chunks using block-level tags."""

    BLOCK_TAGS = [
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "p",
        "div",
        "section",
        "article",
        "li",
        "blockquote",
        "pre",
    ]

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []

        parts = self._extract_block_parts(text)
        if not parts:
            plain = self._to_plain_text(text)
            return [plain] if plain else []

        return self._merge(parts)

    def _extract_block_parts(self, text: str) -> list[str]:
        tags = "|".join(self.BLOCK_TAGS)
        pattern = re.compile(
            rf"<(?P<tag>{tags})\b[^>]*>(?P<content>.*?)</(?P=tag)>",
            re.IGNORECASE | re.DOTALL,
        )

        parts: list[str] = []
        for match in pattern.finditer(text):
            content = self._to_plain_text(match.group("content"))
            if content:
                parts.append(content)
        return parts

    @staticmethod
    def _to_plain_text(fragment: str) -> str:
        no_scripts = re.sub(
            r"<(script|style)\b[^>]*>.*?</\1>", " ", fragment, flags=re.IGNORECASE | re.DOTALL
        )
        no_tags = re.sub(r"<[^>]+>", " ", no_scripts)
        unescaped = html.unescape(no_tags)
        normalized = re.sub(r"\s+", " ", unescaped).strip()
        normalized = re.sub(r"\s+([.,!?;:])", r"\1", normalized)
        return normalized

    def _merge(self, parts: list[str]) -> list[str]:
        chunks: list[str] = []
        current = ""

        for part in parts:
            candidate = f"{current}\n\n{part}" if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(current)

            if len(part) <= self.chunk_size:
                current = part
                continue

            # hard split oversized part
            step = (
                self.chunk_size - self.chunk_overlap if self.chunk_overlap > 0 else self.chunk_size
            )
            for i in range(0, len(part), step):
                chunks.append(part[i : i + self.chunk_size])
            current = ""

        if current:
            chunks.append(current)

        if self.chunk_overlap <= 0 or len(chunks) < 2:
            return chunks

        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-self.chunk_overlap :]
            overlapped.append(tail + chunks[i])
        return overlapped
