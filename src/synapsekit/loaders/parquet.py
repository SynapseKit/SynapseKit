from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from .base import Document


class ParquetLoader:
    """Load documents from an Apache Parquet file.

    Each row in the file becomes a :class:`Document`. Text is assembled
    from the columns listed in ``text_fields`` (in order), or from every
    column when ``text_fields`` is ``None``.
    """

    def __init__(
        self,
        path: str,
        text_fields: list[str] | None = None,
        limit: int | None = None,
    ) -> None:
        if not path:
            raise ValueError("path must be provided")

        self._path = path
        self._text_fields = text_fields
        self._limit = limit

    def load(self) -> list[Document]:
        if not Path(self._path).exists():
            raise FileNotFoundError(f"Parquet file not found: {self._path!r}")

        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("pyarrow required: pip install synapsekit[parquet]") from None

        table = pq.read_table(self._path)
        rows: list[dict[str, Any]] = table.to_pylist()

        if not rows:
            return []

        if self._limit is not None:
            rows = rows[: self._limit]

        docs: list[Document] = []
        for idx, row in enumerate(rows):
            text = self._build_text(row)
            if not text:
                continue
            docs.append(
                Document(
                    text=text,
                    metadata={"source": self._path, "row": idx},
                )
            )

        return docs

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)

    def _build_text(self, row: dict[str, Any]) -> str:
        if self._text_fields is not None:
            parts = []
            for field in self._text_fields:
                val = row.get(field)
                if val not in (None, ""):
                    parts.append(str(row.get(field, "")))
            return " ".join(parts)
        return " ".join(str(v) for v in row.values() if v is not None and v != "")
