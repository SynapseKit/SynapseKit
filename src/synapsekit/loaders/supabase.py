from __future__ import annotations

import os
from typing import Any

from .base import Document


class SupabaseLoader:
    """Load data from a Supabase table."""

    def __init__(
        self,
        table: str,
        supabase_url: str | None = None,
        supabase_key: str | None = None,
        text_columns: list[str] | None = None,
        metadata_columns: list[str] | None = None,
    ) -> None:
        self._table = table
        self._supabase_url = supabase_url or os.environ.get("SUPABASE_URL")
        self._supabase_key = supabase_key or os.environ.get("SUPABASE_KEY")
        self._text_columns = text_columns
        self._metadata_columns = metadata_columns

        if not self._supabase_url or not self._supabase_key:
            raise ValueError("supabase_url and supabase_key are required (or set via environment)")

    def load(self) -> list[Document]:
        try:
            from supabase import Client, create_client
        except ImportError:
            raise ImportError("supabase required: pip install synapsekit[supabase]") from None

        client: Client = create_client(self._supabase_url, self._supabase_key)

        columns = "*"
        if self._text_columns is not None and self._metadata_columns is not None:
            columns = ",".join(self._text_columns + self._metadata_columns)
        elif self._text_columns is not None:
            columns = ",".join(self._text_columns)

        response = client.table(self._table).select(columns).execute()
        rows = response.data

        docs: list[Document] = []
        for i, row in enumerate(rows):
            if self._text_columns:
                text = "\n".join(f"{col}: {row.get(col, '')}" for col in self._text_columns)
            else:
                text = "\n".join(f"{k}: {v}" for k, v in row.items())

            metadata: dict[str, Any] = {"table": self._table, "row": i}
            if self._metadata_columns:
                for col in self._metadata_columns:
                    if col in row:
                        metadata[col] = row[col]
            else:
                # Store all row data as metadata if text columns are specified and metadata columns aren't
                if self._text_columns:
                    metadata.update(row)

            docs.append(Document(text=text, metadata=metadata))

        return docs
