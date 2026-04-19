from __future__ import annotations

import asyncio
from typing import Any

from .base import Document


class AirtableLoader:
    """Load records from an Airtable table as Documents."""

    def __init__(
        self,
        api_key: str,
        base_id: str,
        table_name: str,
        text_fields: list[str] | None = None,
        limit: int | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must be provided")
        if not base_id:
            raise ValueError("base_id must be provided")
        if not table_name:
            raise ValueError("table_name must be provided")

        self._api_key = api_key
        self._base_id = base_id
        self._table_name = table_name
        self._text_fields = text_fields
        self._limit = limit

    def load(self) -> list[Document]:
        try:
            from pyairtable import Table
        except ImportError:
            raise ImportError("pyairtable required: pip install synapsekit[airtable]") from None

        try:
            table = Table(self._api_key, self._base_id, self._table_name)
            # NOTE: table.all() handles pagination internally
            records = table.all(max_records=self._limit) if self._limit else table.all()
        except Exception as e:
            raise RuntimeError(f"Failed to load Airtable records: {e}") from e

        if not records:
            return []

        docs: list[Document] = []
        for record in records:
            # Airtable stores user data under "fields"
            fields = record.get("fields", {})
            if not isinstance(fields, dict):
                fields = {}

            text = self._build_text(fields)
            if not text:
                continue

            metadata: dict[str, Any] = {
                "source": "airtable",
                "record_id": record.get("id"),
                "created_time": record.get("createdTime"),
                **fields,
            }
            docs.append(Document(text=text, metadata=metadata))

        return docs

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)

    def _build_text(self, fields: dict[str, Any]) -> str:
        parts: list[str] = []
        if self._text_fields is not None:
            for field in self._text_fields:
                val = fields.get(field)
                if val not in (None, ""):
                    if isinstance(val, (list, dict)):
                        val = str(val)
                    parts.append(str(val))
            return " ".join(parts)

        for val in fields.values():
            if val not in (None, ""):
                if isinstance(val, (list, dict)):
                    val = str(val)
                parts.append(str(val))
        return " ".join(parts)
