from __future__ import annotations

import os
from typing import Any

from .._json import dumps as _json_dumps
from .._json import loads as _json_loads
from ._cache import AsyncLRUCache


class FilesystemLLMCache:
    """Persistent LLM cache backed by JSON files on disk.

    Each cache entry is stored as a separate ``.json`` file in the cache directory.
    Uses the same ``make_key`` logic as :class:`AsyncLRUCache`.

    Usage::

        from synapsekit.llm._filesystem_cache import FilesystemLLMCache

        cache = FilesystemLLMCache(".synapsekit_cache")
        cache.put(key, value)
        cached = cache.get(key)
    """

    __slots__ = ("_cache_dir", "hits", "misses")

    make_key = staticmethod(AsyncLRUCache.make_key)

    def __init__(self, cache_dir: str = ".synapsekit_cache") -> None:
        self._cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.hits: int = 0
        self.misses: int = 0

    def _path_for(self, key: str) -> str:
        return os.path.join(self._cache_dir, f"{key}.json")

    def get(self, key: str) -> Any | None:
        path = self._path_for(key)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.hits += 1
                return _json_loads(f.read())
        self.misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        path = self._path_for(key)
        with open(path, "w", encoding="utf-8") as f:
            f.write(_json_dumps(value))

    def clear(self) -> None:
        for name in os.listdir(self._cache_dir):
            if name.endswith(".json"):
                os.remove(os.path.join(self._cache_dir, name))

    def __len__(self) -> int:
        return sum(1 for name in os.listdir(self._cache_dir) if name.endswith(".json"))
