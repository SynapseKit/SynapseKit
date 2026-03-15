"""Redis-backed LLM cache."""

from __future__ import annotations

import json
from typing import Any

from ._cache import AsyncLRUCache


class RedisLLMCache:
    """Persistent LLM cache backed by Redis.

    Uses the same ``make_key`` logic as :class:`AsyncLRUCache`.

    Usage::

        from synapsekit.llm._redis_cache import RedisLLMCache

        cache = RedisLLMCache(url="redis://localhost:6379")
        cache.put(key, value)
        cached = cache.get(key)

    Requires ``redis``: ``pip install synapsekit[redis]``
    """

    make_key = staticmethod(AsyncLRUCache.make_key)

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        prefix: str = "synapsekit:llm:",
        ttl: int | None = None,
    ) -> None:
        try:
            import redis
        except ImportError:
            raise ImportError(
                "redis required for RedisLLMCache: pip install synapsekit[redis]"
            ) from None

        self._client = redis.Redis.from_url(url, decode_responses=True)
        self._prefix = prefix
        self._ttl = ttl
        self.hits: int = 0
        self.misses: int = 0

    def _full_key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def get(self, key: str) -> Any | None:
        raw = self._client.get(self._full_key(key))
        if raw is not None:
            self.hits += 1
            return json.loads(raw)
        self.misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        full_key = self._full_key(key)
        self._client.set(full_key, json.dumps(value))
        if self._ttl is not None:
            self._client.expire(full_key, self._ttl)

    def clear(self) -> None:
        cursor = "0"
        while cursor:
            cursor, keys = self._client.scan(cursor=cursor, match=f"{self._prefix}*", count=100)
            if keys:
                self._client.delete(*keys)

    def __len__(self) -> int:
        count = 0
        cursor = "0"
        while cursor:
            cursor, keys = self._client.scan(cursor=cursor, match=f"{self._prefix}*", count=100)
            count += len(keys)
        return count
