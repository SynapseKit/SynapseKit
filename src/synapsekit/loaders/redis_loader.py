from __future__ import annotations

import asyncio
import json
from typing import Any

from .base import Document


class RedisLoader:
    """Load documents from Redis keys matching a pattern.

    Supports three value types:
    - ``"string"`` — plain string values via ``GET``
    - ``"hash"``   — hash values via ``HGETALL``
    - ``"json"``   — JSON-encoded strings via ``GET`` + ``json.loads``
    """

    _SUPPORTED_TYPES = {"string", "hash", "json"}

    def __init__(
        self,
        url: str,
        pattern: str = "*",
        value_type: str = "string",
        limit: int | None = None,
    ) -> None:
        if not url:
            raise ValueError("url must be provided")
        if value_type not in self._SUPPORTED_TYPES:
            raise ValueError(
                f"value_type must be one of {self._SUPPORTED_TYPES!r}, got {value_type!r}"
            )

        self._url = url
        self._pattern = pattern
        self._value_type = value_type
        self._limit = limit

    def load(self) -> list[Document]:
        try:
            from redis import Redis
        except ImportError:
            raise ImportError("redis required: pip install synapsekit[redis]") from None

        client = Redis.from_url(self._url, decode_responses=True)

        keys: list[str] = client.keys(self._pattern)
        if self._limit is not None:
            keys = keys[: self._limit]

        docs: list[Document] = []
        for key in keys:
            text = self._fetch_text(client, key)
            if not text:
                continue
            docs.append(Document(text=text, metadata={"source": "redis", "key": key}))

        return docs

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)

    def _fetch_text(self, client: Any, key: str) -> str:
        if self._value_type == "string":
            return self._fetch_string(client, key)
        if self._value_type == "hash":
            return self._fetch_hash(client, key)
        return self._fetch_json(client, key)

    def _fetch_string(self, client: Any, key: str) -> str:
        value = client.get(key)
        if value is None:
            return ""
        return str(value)

    def _fetch_hash(self, client: Any, key: str) -> str:
        data: dict[str, str] = client.hgetall(key)
        if not data:
            return ""
        return " ".join(f"{k}: {v}" for k, v in data.items())

    def _fetch_json(self, client: Any, key: str) -> str:
        raw = client.get(key)
        if raw is None:
            return ""
        try:
            data = json.loads(raw)
            return json.dumps(data)
        except (json.JSONDecodeError, TypeError):
            return str(raw)
