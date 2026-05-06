from __future__ import annotations

from collections import OrderedDict
from typing import Any

from .._json import dumps_bytes as _json_dumps_bytes

try:
    from .._rust_core import fast_cache_key as _rust_cache_key
except ImportError:
    _rust_cache_key = None

try:
    from xxhash import xxh3_128_hexdigest as _xxh3_hex
except ImportError:
    _xxh3_hex = None  # type: ignore[assignment]


class AsyncLRUCache:
    """Simple LRU cache backed by an ``OrderedDict``."""

    __slots__ = ("_cache", "_maxsize", "hits", "misses")

    def __init__(self, maxsize: int = 128) -> None:
        self._maxsize = maxsize
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self.hits: int = 0
        self.misses: int = 0

    @staticmethod
    def make_key(
        model: str,
        prompt_or_messages: str | list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Deterministic cache key from request parameters.

        Uses Rust BLAKE3 → xxhash → sha256 fallback chain (fastest available).
        """
        if _rust_cache_key is not None:
            return _rust_cache_key(model, prompt_or_messages, temperature, max_tokens)  # type: ignore[no-any-return]
        payload = _json_dumps_bytes(
            {
                "model": model,
                "input": prompt_or_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        if _xxh3_hex is not None:
            return _xxh3_hex(payload)  # type: ignore[no-any-return]
        import hashlib

        return hashlib.sha256(payload).hexdigest()

    def get(self, key: str) -> Any | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]
        self.misses += 1
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
