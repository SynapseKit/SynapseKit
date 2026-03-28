from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from synapsekit.llm._cache import AsyncLRUCache

logger = logging.getLogger(__name__)

try:
    import aiomcache

    AIOMCACHE_AVAILABLE = True
except ImportError:
    AIOMCACHE_AVAILABLE = False


def _run_sync(coro: Any) -> Any:
    """Helper to run coroutines synchronously in the cache methods."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio

            nest_asyncio.apply()
            return asyncio.run_coroutine_threadsafe(coro, loop).result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


class MemcachedCacheBackend(AsyncLRUCache):
    """LLM Cache backend using Memcached.

    Expects a Memcached server address (default: `127.0.0.1:11211`).
    Values are stored as JSON serialized strings.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11211,
        ttl_seconds: int = 0,
        **kwargs: Any,
    ) -> None:
        if not AIOMCACHE_AVAILABLE:
            raise ImportError(
                "aiomcache is required to use MemcachedCacheBackend. "
                "Install it with `pip install aiomcache` or `pip install synapsekit[memcached]`."
            )

        # Skip AsyncLRUCache init as we don't need in-memory storage,
        # but we maintain hit/miss counters.
        self.hits: int = 0
        self.misses: int = 0
        self.host = host
        self.port = port
        self.ttl_seconds = ttl_seconds

        self._client = aiomcache.Client(host, port, **kwargs)

    def get(self, key: str) -> Any | None:
        try:
            value = _run_sync(self._client.get(key.encode("utf-8")))
            if value is not None:
                self.hits += 1
                return json.loads(value.decode("utf-8"))
        except Exception as e:
            logger.error(f"Memcached cache get error for key {key}: {e}")

        self.misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        try:
            serialized_value = json.dumps(value).encode("utf-8")
            _run_sync(
                self._client.set(
                    key.encode("utf-8"),
                    serialized_value,
                    exptime=self.ttl_seconds,
                )
            )
        except TypeError as e:
            logger.error(f"Failed to serialize value for caching: {e}")
        except Exception as e:
            logger.error(f"Memcached cache put error for key {key}: {e}")

    def clear(self) -> None:
        """Clear all items from the Memcached server."""
        try:
            _run_sync(self._client.flush_all())
        except Exception as e:
            logger.error(f"Memcached cache clear error: {e}")

    def __len__(self) -> int:
        """Warning: __len__() is generally not well-supported by Memcached without
        querying stats. Returning 0 to satisfy the interface.
        """
        logger.warning(
            "__len__() called on MemcachedCacheBackend, but it is not accurately supported "
            "by Memcached. Returning 0."
        )
        return 0
