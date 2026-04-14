from __future__ import annotations

import asyncio
import csv
import os
from typing import Any

from .base import Document


class TSVLoader:
    """Load a TSV file, one Document per row. Columns become metadata."""

    def __init__(
        self,
        path: str,
        text_column: str | None = None,
        encoding: str = "utf-8",
    ) -> None:
        self._path = path
        self._text_column = text_column
        self._encoding = encoding

    def load(self) -> list[Document]:
        if not os.path.exists(self._path):
            raise FileNotFoundError(f"TSV file not found: {self._path}")
        docs = []
        with open(self._path, encoding=self._encoding, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            if reader.fieldnames is None:
                return []
            for i, row in enumerate(reader):
                if not any(row.values()):
                    continue
                if self._text_column:
                    text = str(row.get(self._text_column) or "")
                    meta: dict[str, Any] = {
                        k: str(v).strip() if v is not None else ""
                        for k, v in row.items()
                        if k != self._text_column
                    }
                else:
                    text = " ".join(str(v).strip() for v in row.values() if v is not None)
                    meta = {k: str(v).strip() if v is not None else "" for k, v in row.items()}
                meta["source"] = self._path
                meta["row"] = i
                docs.append(Document(text=text, metadata=meta))
        return docs

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)
