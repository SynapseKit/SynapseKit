from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from .base import Document


class MongoDBLoader:
    """Load documents from a MongoDB collection."""

    def __init__(
        self,
        connection_string: str,
        database: str,
        collection: str,
        query_filter: Mapping[str, Any] | None = None,
        text_fields: list[str] | None = None,
        metadata_fields: list[str] | None = None,
    ) -> None:
        if not connection_string:
            raise ValueError("connection_string must be provided")
        if not database:
            raise ValueError("database must be provided")
        if not collection:
            raise ValueError("collection must be provided")

        self._connection_string = connection_string
        self._database = database
        self._collection = collection
        self._query_filter = dict(query_filter or {})
        self._text_fields = text_fields
        self._metadata_fields = metadata_fields

    def load(self) -> list[Document]:
        try:
            from pymongo import MongoClient
        except ImportError:
            raise ImportError("pymongo required: pip install synapsekit[mongodb]") from None

        client = MongoClient(self._connection_string)
        try:
            coll = client[self._database][self._collection]
            projection = self._build_projection()
            if projection is None:
                rows = list(coll.find(self._query_filter))
            else:
                rows = list(coll.find(self._query_filter, projection))
        finally:
            client.close()

        docs: list[Document] = []
        for idx, row in enumerate(rows):
            text = self._build_text(row)
            metadata = self._build_metadata(row, idx)
            docs.append(Document(text=text, metadata=metadata))

        return docs

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)

    def _build_projection(self) -> dict[str, int] | None:
        fields: set[str] = set()
        if self._text_fields:
            fields.update(self._text_fields)
        if self._metadata_fields:
            fields.update(self._metadata_fields)

        if not fields:
            return None

        # Keep _id by default so callers can include it in metadata if desired.
        fields.add("_id")
        return {field: 1 for field in fields}

    def _build_text(self, row: Mapping[str, Any]) -> str:
        if self._text_fields:
            parts = []
            for field in self._text_fields:
                value = row.get(field, "")
                parts.append("" if value is None else str(value))
            return "\n".join(parts)

        return "\n".join(f"{key}: {value}" for key, value in row.items())

    def _build_metadata(self, row: Mapping[str, Any], idx: int) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "source": "mongodb",
            "database": self._database,
            "collection": self._collection,
            "row": idx,
            "query": dict(self._query_filter),
        }

        if self._metadata_fields:
            for field in self._metadata_fields:
                if field in row:
                    metadata[field] = row[field]
        else:
            metadata.update(row)

        return metadata
