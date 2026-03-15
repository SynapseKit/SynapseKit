from __future__ import annotations

import os
import re

from .base import Document


class MarkdownLoader:
    """Load text from a Markdown file.

    Optionally strips YAML frontmatter (``---`` delimited blocks at the start).
    No external dependencies required.
    """

    def __init__(self, path: str, strip_frontmatter: bool = True) -> None:
        self._path = path
        self._strip_frontmatter = strip_frontmatter

    def load(self) -> list[Document]:
        if not os.path.exists(self._path):
            raise FileNotFoundError(f"File not found: {self._path}")

        with open(self._path, encoding="utf-8") as f:
            text = f.read()

        if self._strip_frontmatter:
            text = re.sub(r"\A---\n.*?\n---\n?", "", text, count=1, flags=re.DOTALL)

        return [Document(text=text, metadata={"source": self._path})]
