from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

from .base import Document


class ObsidianLoader:
    """Load Markdown notes from an Obsidian vault directory."""

    # Best-effort wikilink extraction (does not handle nested brackets)
    _WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
    # Best-effort tag extraction; may match tags inside code/URLs
    _TAG_RE = re.compile(r"(?<!\w)#([A-Za-z0-9_-]+)")

    def __init__(self, path: str, encoding: str = "utf-8", recursive: bool = True) -> None:
        self._path = path
        self._encoding = encoding
        self._recursive = recursive

    def load(self) -> list[Document]:
        if not os.path.exists(self._path):
            raise FileNotFoundError(f"Path not found: {self._path}")
        if not os.path.isdir(self._path):
            raise NotADirectoryError(f"Path is not a directory: {self._path}")

        root = Path(self._path)
        iterator = root.rglob("*.md") if self._recursive else root.glob("*.md")
        docs: list[Document] = []

        for file_path in sorted(iterator):
            if not file_path.is_file():
                continue
            try:
                with open(file_path, encoding=self._encoding, errors="ignore") as f:
                    content = f.read()
            except FileNotFoundError:
                continue

            frontmatter, text = self._extract_frontmatter(content)
            if text:
                links = self._extract_wikilinks(text)
                tags = self._extract_tags(text)
            else:
                links, tags = [], []
            if not text and not frontmatter:
                continue

            docs.append(
                Document(
                    text=text,
                    metadata={
                        "source": str(file_path),
                        "frontmatter": frontmatter,
                        "tags": tags,
                        "links": links,
                    },
                )
            )

        return docs

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)

    def _extract_frontmatter(self, content: str) -> tuple[dict[str, object], str]:
        normalized_content = content.lstrip()
        if not normalized_content.startswith("---"):
            return {}, content

        lines = normalized_content.splitlines(keepends=True)
        if not lines or lines[0].strip() != "---":
            return {}, content

        end_index = None
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                end_index = i
                break

        if end_index is None:
            return {}, content

        raw_frontmatter = "".join(lines[1:end_index])
        text_without_frontmatter = "".join(lines[end_index + 1 :]).strip()
        return self._parse_frontmatter(raw_frontmatter), text_without_frontmatter

    def _parse_frontmatter(self, raw_frontmatter: str) -> dict[str, object]:
        # Simple YAML parser (does not support nested or multiline values)
        parsed: dict[str, object] = {}
        for line in raw_frontmatter.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue

            if value.startswith("[") and value.endswith("]"):
                inner = value[1:-1].strip()
                items = [item.strip().strip("\"'") for item in inner.split(",") if item.strip()]
                parsed[key] = items
            else:
                parsed[key] = value.strip("\"'")
        return parsed

    def _extract_wikilinks(self, text: str) -> list[str]:
        links: list[str] = []
        seen = set()
        for match in self._WIKILINK_RE.findall(text):
            page = match.split("|", 1)[0].strip()
            if page and page not in seen:
                seen.add(page)
                links.append(page)
        return links

    def _extract_tags(self, text: str) -> list[str]:
        tags: list[str] = []
        seen = set()
        for tag in self._TAG_RE.findall(text):
            if tag not in seen:
                seen.add(tag)
                tags.append(tag)
        return tags
