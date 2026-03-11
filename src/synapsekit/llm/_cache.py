from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import Any


class AsyncLRUCache:
    """Simple LRU cache backed by an ``OrderedDict``."""

    def __init__(self, maxsize: int = 128) -> None:
        self._maxsize = maxsize
        self._cache: OrderedDict[str, Any] = OrderedDict()

    @staticmethod
    def make_key(
        model: str,
        prompt_or_messages: str | list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Deterministic cache key from request parameters."""
        payload = json.dumps(
            {
                "model": model,
                "input": prompt_or_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> Any | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, value: Any) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)
