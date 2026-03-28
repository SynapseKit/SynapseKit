from __future__ import annotations

import json
import logging
from typing import Any

from synapsekit.llm._cache import AsyncLRUCache

logger = logging.getLogger(__name__)

try:
    import boto3
    from botocore.exceptions import ClientError

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class DynamoDBCacheBackend(AsyncLRUCache):
    """LLM Cache backend using AWS DynamoDB.

    Expects a DynamoDB table with a primary partition key (default: `cache_key`)
    and columns `value` and optionally `ttl` for DynamoDB time-to-live functionality.
    """

    def __init__(
        self,
        table_name: str,
        region_name: str | None = None,
        ttl_seconds: int | None = None,
        partition_key: str = "cache_key",
        **kwargs: Any,
    ) -> None:
        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required to use DynamoDBCacheBackend. "
                "Install it with `pip install boto3` or `pip install synapsekit[dynamodb]`."
            )

        # Skip AsyncLRUCache init as we don't need in-memory storage,
        # but we maintain hit/miss counters.
        self.hits: int = 0
        self.misses: int = 0
        self.table_name = table_name
        self.partition_key = partition_key
        self.ttl_seconds = ttl_seconds

        client_kwargs = {}
        if region_name:
            client_kwargs["region_name"] = region_name
        client_kwargs.update(kwargs)

        self._dynamodb = boto3.resource("dynamodb", **client_kwargs)
        self._table = self._dynamodb.Table(self.table_name)

    def get(self, key: str) -> Any | None:
        try:
            response = self._table.get_item(Key={self.partition_key: key})
            item = response.get("Item")
            if item and "value" in item:
                self.hits += 1
                return json.loads(item["value"])
        except ClientError as e:
            logger.error(f"DynamoDB cache get error for key {key}: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode cached value for key {key}: {e}")

        self.misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        try:
            item: dict[str, Any] = {
                self.partition_key: key,
                "value": json.dumps(value),
            }
            if self.ttl_seconds is not None:
                import time

                item["ttl"] = int(time.time()) + int(self.ttl_seconds)

            self._table.put_item(Item=item)
        except ClientError as e:
            logger.error(f"DynamoDB cache put error for key {key}: {e}")
        except TypeError as e:
            logger.error(f"Failed to serialize value for caching: {e}")

    def clear(self) -> None:
        """Warning: clear() is generally not recommended for DynamoDB as it requires
        scanning and deleting items one by one, or deleting and recreating the table.
        This implementation relies on TTL for eviction.
        """
        logger.warning(
            "clear() called on DynamoDBCacheBackend, but it is not implemented "
            "due to DynamoDB limitations. Rely on TTL for item eviction."
        )

    def __len__(self) -> int:
        """Approximate item count via table metadata."""
        try:
            self._table.reload()
            return int(self._table.item_count)
        except ClientError:
            return 0
