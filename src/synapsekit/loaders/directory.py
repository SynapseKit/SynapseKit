from __future__ import annotations

import os
from glob import glob
from typing import Any

from .base import Document


def _loader_for(path: str) -> Any:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        from .pdf import PDFLoader

        return PDFLoader(path)
    elif ext in (".html", ".htm"):
        from .html import HTMLLoader

        return HTMLLoader(path)
    elif ext == ".csv":
        from .csv import CSVLoader

        return CSVLoader(path)
    elif ext == ".json":
        from .json_loader import JSONLoader

        return JSONLoader(path)
    else:
        from .text import TextLoader

        return TextLoader(path)


class DirectoryLoader:
    """Load all matching files in a directory, delegating to per-format loaders."""

    def __init__(
        self,
        path: str,
        glob_pattern: str = "**/*.*",
        recursive: bool = True,
    ) -> None:
        self._path = path
        self._glob = glob_pattern
        self._recursive = recursive

    def load(self) -> list[Document]:
        pattern = os.path.join(self._path, self._glob)
        paths = glob(pattern, recursive=self._recursive)
        docs: list[Document] = []
        for p in sorted(paths):
            if os.path.isfile(p):
                try:
                    loader = _loader_for(p)
                    docs.extend(loader.load())
                except Exception:
                    pass
        return docs
