from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from .base import Document


class BigQueryLoader:
    """Load rows from Google BigQuery via SQL or table reference."""

    def __init__(
        self,
        query: str | None = None,
        table: str | None = None,
        dataset: str | None = None,
        project: str | None = None,
        text_fields: list[str] | None = None,
        metadata_fields: list[str] | None = None,
        client: Any | None = None,
    ) -> None:
        if not query and not table:
            raise ValueError("Either query or table must be provided")
        if query and table:
            raise ValueError("Provide either query or table, not both")
        if table and not dataset and "." not in table:
            raise ValueError("dataset must be provided when table is not fully qualified")

        self._query = query
        self._table = table
        self._dataset = dataset
        self._project = project
        self._text_fields = text_fields
        self._metadata_fields = metadata_fields
        self._client = client

    def load(self) -> list[Document]:
        client = self._client
        if client is None:
            try:
                from google.cloud import bigquery
            except ImportError:
                raise ImportError(
                    "google-cloud-bigquery required: pip install synapsekit[bigquery]"
                ) from None

            client = bigquery.Client(project=self._project) if self._project else bigquery.Client()

        query = self._query if self._query is not None else self._build_table_query()
        result = client.query(query).result()

        docs: list[Document] = []
        for idx, row in enumerate(result):
            row_data = self._row_to_dict(row)
            if row_data is None:
                continue

            text = self._build_text(row_data)
            metadata = self._build_metadata(row_data, query, idx)
            docs.append(Document(text=text, metadata=metadata))

        return docs

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)

    def _build_table_query(self) -> str:
        if self._table is None:
            raise ValueError("table must be provided when query is not set")

        if "." in self._table:
            table_ref = self._table
        else:
            if self._dataset is None:
                raise ValueError("dataset must be provided when table is not fully qualified")

            table_ref = f"{self._dataset}.{self._table}"
            if self._project:
                table_ref = f"{self._project}.{table_ref}"

        return f"SELECT * FROM `{table_ref}`"

    def _row_to_dict(self, row: Any) -> dict[str, Any] | None:
        if isinstance(row, Mapping):
            return dict(row)

        if hasattr(row, "items"):
            try:
                return dict(row.items())
            except Exception:
                return None

        return None

    def _build_text(self, row: Mapping[str, Any]) -> str:
        if self._text_fields:
            parts = []
            for field in self._text_fields:
                value = row.get(field, "")
                safe_value = "" if value is None else str(value)
                parts.append(f"{field}: {safe_value}")
            return "\n".join(parts)

        return "\n".join(f"{key}: {value}" for key, value in row.items())

    def _build_metadata(self, row: Mapping[str, Any], query: str, row_index: int) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "source": "bigquery",
            "query": query,
            "row": row_index,
        }

        if self._project:
            metadata["project"] = self._project
        if self._dataset:
            metadata["dataset"] = self._dataset
        if self._table:
            metadata["table"] = self._table

        if self._metadata_fields:
            for field in self._metadata_fields:
                if field in row:
                    metadata[field] = row[field]
        else:
            metadata.update(row)

        return metadata
