from __future__ import annotations

import csv
from typing import List, Optional

from .base import Document


class CSVLoader:
    """Load a CSV file, one Document per row. Columns become metadata."""

    def __init__(
        self,
        path: str,
        text_column: Optional[str] = None,
        encoding: str = "utf-8",
    ) -> None:
        self._path = path
        self._text_column = text_column
        self._encoding = encoding

    def load(self) -> List[Document]:
        docs = []
        with open(self._path, encoding=self._encoding, newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if self._text_column:
                    text = str(row.get(self._text_column, ""))
                    meta = {k: v for k, v in row.items() if k != self._text_column}
                else:
                    text = " ".join(str(v) for v in row.values())
                    meta = dict(row)
                meta["source"] = self._path
                meta["row"] = i
                docs.append(Document(text=text, metadata=meta))
        return docs
