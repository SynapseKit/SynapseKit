from __future__ import annotations

import os
from typing import List

from .base import Document


class StringLoader:
    """Wrap a raw string as a loadable document."""

    def __init__(self, text: str, metadata: dict | None = None) -> None:
        self._text = text
        self.metadata = metadata or {}

    def load(self) -> List[Document]:
        return [Document(text=self._text, metadata=self.metadata)]


class TextLoader:
    """Load plain text from a file path."""

    def __init__(self, path: str, encoding: str = "utf-8") -> None:
        self._path = path
        self._encoding = encoding

    def load(self) -> List[Document]:
        if not os.path.exists(self._path):
            raise FileNotFoundError(f"File not found: {self._path}")
        with open(self._path, encoding=self._encoding) as f:
            text = f.read()
        return [Document(text=text, metadata={"source": self._path})]
