from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from .base import Document

_SUPPORTED_OBJECT_TYPES = {"contacts", "deals", "tickets"}


class HubSpotLoader:
    """Load CRM objects from HubSpot."""

    def __init__(
        self,
        object_type: str,
        access_token: str | None = None,
        text_fields: list[str] | None = None,
        metadata_fields: list[str] | None = None,
        limit: int = 100,
        client: Any | None = None,
    ) -> None:
        if not isinstance(object_type, str) or not object_type.strip():
            raise ValueError("object_type must be provided")

        normalized_object_type = object_type.strip().lower()
        if normalized_object_type not in _SUPPORTED_OBJECT_TYPES:
            raise ValueError("object_type must be one of: contacts, deals, tickets")

        if client is None and not access_token:
            raise ValueError("access_token is required unless client is provided")

        if limit <= 0:
            raise ValueError("limit must be greater than 0")

        self._object_type = normalized_object_type
        self._access_token = access_token
        self._text_fields = text_fields
        self._metadata_fields = metadata_fields
        self._limit = limit
        self._client = client

    def load(self) -> list[Document]:
        client = self._client
        if client is None:
            try:
                from hubspot import HubSpot
            except ImportError:
                raise ImportError(
                    "hubspot-api-client required: pip install synapsekit[hubspot]"
                ) from None

            client = HubSpot(access_token=self._access_token)

        api = self._resolve_basic_api(client)
        properties = self._build_properties()

        response = api.get_page(limit=self._limit, archived=False, properties=properties)
        results = getattr(response, "results", []) or []

        docs: list[Document] = []
        for index, item in enumerate(results):
            record = self._normalize_record(item)
            if record is None:
                continue

            text = self._build_text(record)
            metadata = self._build_metadata(record, index)
            docs.append(Document(text=text, metadata=metadata))

        return docs

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)

    def _resolve_basic_api(self, client: Any) -> Any:
        if self._object_type == "contacts":
            return client.crm.contacts.basic_api
        if self._object_type == "deals":
            return client.crm.deals.basic_api
        return client.crm.tickets.basic_api

    def _build_properties(self) -> list[str] | None:
        fields: list[str] = []

        if self._text_fields:
            fields.extend(self._text_fields)
        if self._metadata_fields:
            fields.extend(self._metadata_fields)

        if not fields:
            return None

        unique_fields: list[str] = []
        for field in fields:
            if field not in unique_fields:
                unique_fields.append(field)

        return unique_fields

    def _normalize_record(self, item: Any) -> dict[str, Any] | None:
        raw: Any = item

        if hasattr(item, "to_dict") and callable(item.to_dict):
            raw = item.to_dict()

        if not isinstance(raw, Mapping):
            return None

        properties = raw.get("properties")
        merged: dict[str, Any] = dict(properties) if isinstance(properties, Mapping) else {}

        if "id" in raw:
            merged["id"] = raw["id"]
        if "created_at" in raw:
            merged["created_at"] = raw["created_at"]
        if "updated_at" in raw:
            merged["updated_at"] = raw["updated_at"]
        if "archived" in raw:
            merged["archived"] = raw["archived"]

        if not merged:
            merged = dict(raw)

        return merged

    def _build_text(self, record: Mapping[str, Any]) -> str:
        if self._text_fields:
            values: list[str] = []
            for field in self._text_fields:
                value = record.get(field, "")
                safe_value = "" if value is None else str(value)
                values.append(f"{field}: {safe_value}")
            return "\n".join(values)

        return "\n".join(f"{k}: {v}" for k, v in record.items())

    def _build_metadata(self, record: Mapping[str, Any], row: int) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "source": "hubspot",
            "object_type": self._object_type,
            "row": row,
        }

        if self._metadata_fields:
            for field in self._metadata_fields:
                if field in record:
                    metadata[field] = record[field]
        else:
            metadata.update(record)

        return metadata
