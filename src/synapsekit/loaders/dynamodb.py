"""DynamoDBLoader — load items from AWS DynamoDB tables as Documents."""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from typing import Any, Literal

from .base import Document


class DynamoDBLoader:
    """Load items from a DynamoDB table into Documents.

    This loader supports both scan and query operations and converts each
    DynamoDB item into a Document.

    Prerequisites:
        - boto3

    Example::

        loader = DynamoDBLoader(
            table_name="my-table",
            operation="scan",
            text_attributes=["title", "content"],
        )
        docs = loader.load()

    Args:
        table_name: DynamoDB table name.
        operation: Either "scan" or "query".
        scan_kwargs: Extra kwargs forwarded to Table.scan().
        query_kwargs: Extra kwargs forwarded to Table.query().
        text_attributes: Item attributes to concatenate into Document.text.
        metadata_attributes: Item attributes to include in metadata.
        region_name: AWS region (default "us-east-1").
        aws_access_key_id/aws_secret_access_key/aws_session_token: Optional explicit credentials.
        max_items: Optional cap on number of items converted.
    """

    def __init__(
        self,
        table_name: str,
        *,
        operation: Literal["scan", "query"] = "scan",
        scan_kwargs: dict[str, Any] | None = None,
        query_kwargs: dict[str, Any] | None = None,
        text_attributes: list[str] | None = None,
        metadata_attributes: list[str] | None = None,
        region_name: str = "us-east-1",
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        max_items: int | None = None,
    ) -> None:
        if not table_name:
            raise ValueError("table_name must be provided")

        if operation not in {"scan", "query"}:
            raise ValueError("operation must be 'scan' or 'query'")

        self._table_name = table_name
        self._operation: Literal["scan", "query"] = operation
        self._scan_kwargs = dict(scan_kwargs or {})
        self._query_kwargs = dict(query_kwargs or {})
        self._text_attributes = list(text_attributes) if text_attributes else None
        self._metadata_attributes = list(metadata_attributes) if metadata_attributes else None

        self._region_name = region_name
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key
        self._aws_session_token = aws_session_token

        self._max_items = max_items

    def load(self) -> list[Document]:
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 required: pip install synapsekit[dynamodb]") from None

        table = self._get_table(boto3)

        items = self._fetch_items(table)
        docs: list[Document] = []
        for idx, item in enumerate(items):
            if self._max_items is not None and idx >= self._max_items:
                break

            normalized = _normalize_item(item)
            text = self._build_text(normalized)
            metadata = self._build_metadata(normalized, idx)
            docs.append(Document(text=text, metadata=metadata))

        return docs

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)

    def _get_table(self, boto3_module: Any) -> Any:
        kwargs: dict[str, Any] = {"region_name": self._region_name}
        if self._aws_access_key_id:
            kwargs["aws_access_key_id"] = self._aws_access_key_id
        if self._aws_secret_access_key:
            kwargs["aws_secret_access_key"] = self._aws_secret_access_key
        if self._aws_session_token:
            kwargs["aws_session_token"] = self._aws_session_token

        resource = boto3_module.resource("dynamodb", **kwargs)
        return resource.Table(self._table_name)

    def _fetch_items(self, table: Any) -> list[dict[str, Any]]:
        if self._operation == "query" and not self._query_kwargs:
            raise ValueError("query_kwargs must be provided when operation='query'")

        op_name = self._operation
        base_kwargs = dict(self._query_kwargs if op_name == "query" else self._scan_kwargs)

        items: list[dict[str, Any]] = []
        start_key: dict[str, Any] | None = None

        while True:
            kwargs = dict(base_kwargs)
            if start_key is not None:
                kwargs["ExclusiveStartKey"] = start_key

            response = table.query(**kwargs) if op_name == "query" else table.scan(**kwargs)
            page_items = response.get("Items", []) or []
            items.extend(page_items)

            start_key = response.get("LastEvaluatedKey")
            if not start_key:
                break

            if self._max_items is not None and len(items) >= self._max_items:
                break

        return items

    def _build_text(self, item: dict[str, Any]) -> str:
        if self._text_attributes:
            parts: list[str] = []
            for attr in self._text_attributes:
                value = item.get(attr)
                parts.append(_stringify_value(value))
            return "\n".join(parts)

        # Default: show all key/value pairs (stable order for tests/debuggability).
        return "\n".join(f"{k}: {_stringify_value(v)}" for k, v in sorted(item.items()))

    def _build_metadata(self, item: dict[str, Any], idx: int) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "source": "dynamodb",
            "table": self._table_name,
            "row": idx,
            "operation": self._operation,
        }

        if self._metadata_attributes:
            for attr in self._metadata_attributes:
                if attr in item:
                    metadata[attr] = item[attr]
        else:
            metadata.update(item)

        return metadata


def _normalize_item(item: dict[str, Any]) -> dict[str, Any]:
    return {k: _normalize_value(v) for k, v in item.items()}


def _normalize_value(value: Any) -> Any:
    # DynamoDB uses Decimal for numbers.
    if isinstance(value, Decimal):
        if value % 1 == 0:
            return int(value)
        return float(value)

    if isinstance(value, bytes):
        # Make bytes readable/serializable.
        return value.decode("utf-8", errors="replace")

    if isinstance(value, set):
        return sorted(_normalize_value(v) for v in value)

    if isinstance(value, list):
        return [_normalize_value(v) for v in value]

    if isinstance(value, dict):
        return {k: _normalize_value(v) for k, v in value.items()}

    return value


def _stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            return str(value)
    return str(value)
