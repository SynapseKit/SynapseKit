from __future__ import annotations

import json

from .base import Document


class JSONLoader:
    """Load a JSON file (list or single object), configurable text key + metadata keys."""

    def __init__(
        self,
        path: str,
        text_key: str = "text",
        metadata_keys: list[str] | None = None,
        encoding: str = "utf-8",
    ) -> None:
        self._path = path
        self._text_key = text_key
        self._metadata_keys = metadata_keys or []
        self._encoding = encoding

    def load(self) -> list[Document]:
        with open(self._path, encoding=self._encoding) as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = [data]

        docs = []
        for i, item in enumerate(data):
            text = str(item.get(self._text_key, ""))
            meta: dict = {"source": self._path, "index": i}
            for key in self._metadata_keys:
                if key in item:
                    meta[key] = item[key]
            docs.append(Document(text=text, metadata=meta))
        return docs
