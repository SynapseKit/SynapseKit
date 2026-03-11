from __future__ import annotations

import os


class StringLoader:
    """Wrap a raw string as a loadable document."""

    def __init__(self, text: str, metadata: dict | None = None) -> None:
        self._text = text
        self.metadata = metadata or {}

    def load(self) -> str:
        return self._text


class TextLoader:
    """Load plain text from a file path."""

    def __init__(self, path: str, encoding: str = "utf-8") -> None:
        self._path = path
        self._encoding = encoding
        self.metadata: dict = {"source": path}

    def load(self) -> str:
        if not os.path.exists(self._path):
            raise FileNotFoundError(f"File not found: {self._path}")
        with open(self._path, encoding=self._encoding) as f:
            return f.read()
