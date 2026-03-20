"""Redis-backed graph checkpointer."""

from __future__ import annotations

import json
from typing import Any

from .base import BaseCheckpointer

_KEY_PREFIX = "synapsekit:checkpoint:"


class RedisCheckpointer(BaseCheckpointer):
    """Persist graph checkpoints in Redis.

    Requires ``redis`` (``pip install synapsekit[redis]``).

    Usage::

        import redis
        r = redis.Redis()
        cp = RedisCheckpointer(r)
        cp.save("my-graph", 3, {"messages": [...]})
        step, state = cp.load("my-graph")
    """

    def __init__(self, client: Any, *, ttl: int | None = None) -> None:
        """
        Args:
            client: A ``redis.Redis`` instance.
            ttl: Optional TTL in seconds for auto-expiry.
        """
        self._client = client
        self._ttl = ttl

    def _key(self, graph_id: str) -> str:
        return f"{_KEY_PREFIX}{graph_id}"

    def save(self, graph_id: str, step: int, state: dict[str, Any]) -> None:
        """Persist the state at the given step."""
        data = json.dumps({"step": step, "state": state})
        key = self._key(graph_id)
        if self._ttl is not None:
            self._client.setex(key, self._ttl, data)
        else:
            self._client.set(key, data)

    def load(self, graph_id: str) -> tuple[int, dict[str, Any]] | None:
        """Load the most recent checkpoint. Returns ``(step, state)`` or ``None``."""
        raw = self._client.get(self._key(graph_id))
        if raw is None:
            return None
        data = json.loads(raw)
        return data["step"], data["state"]

    def delete(self, graph_id: str) -> None:
        """Remove the checkpoint for the given graph_id."""
        self._client.delete(self._key(graph_id))

    def close(self) -> None:
        """Close the Redis connection."""
        self._client.close()
