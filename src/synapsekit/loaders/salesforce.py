from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from .base import Document


class SalesforceLoader:
    """Load Salesforce records via SOQL query."""

    def __init__(
        self,
        soql_query: str,
        username: str | None = None,
        password: str | None = None,
        security_token: str | None = None,
        domain: str = "login",
        text_fields: list[str] | None = None,
        metadata_fields: list[str] | None = None,
        client: Any | None = None,
    ) -> None:
        if not soql_query:
            raise ValueError("soql_query must be provided")

        if client is None and (not username or not password or not security_token):
            raise ValueError(
                "username, password, and security_token are required unless client is provided"
            )

        self._soql_query = soql_query
        self._username = username
        self._password = password
        self._security_token = security_token
        self._domain = domain
        self._text_fields = text_fields
        self._metadata_fields = metadata_fields
        self._client = client

    def load(self) -> list[Document]:
        client = self._client
        if client is None:
            try:
                from simple_salesforce import Salesforce
            except ImportError:
                raise ImportError(
                    "simple-salesforce required: pip install synapsekit[salesforce]"
                ) from None

            client = Salesforce(
                username=self._username,
                password=self._password,
                security_token=self._security_token,
                domain=self._domain,
            )

        result = client.query_all(self._soql_query)
        records = result.get("records", []) if isinstance(result, Mapping) else []

        docs: list[Document] = []
        for idx, raw_record in enumerate(records):
            if not isinstance(raw_record, Mapping):
                continue

            clean_record = {k: v for k, v in raw_record.items() if k != "attributes"}
            text = self._build_text(clean_record)
            metadata = self._build_metadata(clean_record, raw_record.get("attributes"), idx)
            docs.append(Document(text=text, metadata=metadata))

        return docs

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)

    def _build_text(self, record: Mapping[str, Any]) -> str:
        if self._text_fields:
            parts = []
            for field in self._text_fields:
                value = record.get(field, "")
                safe_value = "" if value is None else str(value)
                parts.append(f"{field}: {safe_value}")
            return "\n".join(parts)

        return "\n".join(f"{k}: {v}" for k, v in record.items())

    def _build_metadata(
        self,
        record: Mapping[str, Any],
        attributes: Any,
        row: int,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "source": "salesforce",
            "query": self._soql_query,
            "row": row,
        }

        if isinstance(attributes, Mapping) and "type" in attributes:
            metadata["object"] = attributes["type"]

        if self._metadata_fields:
            for field in self._metadata_fields:
                if field in record:
                    metadata[field] = record[field]
        else:
            metadata.update(record)

        return metadata
