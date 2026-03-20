"""Persistent conversation memory backed by Redis."""

from __future__ import annotations

import json


class RedisConversationMemory:
    """Persistent conversation memory using Redis.

    Messages survive process restarts. Supports multiple conversations
    via ``conversation_id``. Requires ``redis`` package (``pip install synapsekit[redis]``).

    Usage::

        memory = RedisConversationMemory(url="redis://localhost:6379", conversation_id="user-1")
        memory.add("user", "Hello!")
        memory.add("assistant", "Hi there!")
        messages = memory.get_messages()  # persisted in Redis

    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        conversation_id: str = "default",
        window: int | None = None,
        prefix: str = "synapsekit:memory:",
    ) -> None:
        try:
            import redis
        except ImportError:
            raise ImportError("redis package required: pip install synapsekit[redis]") from None

        self._url = url
        self._conversation_id = conversation_id
        self._window = window
        self._prefix = prefix
        self._client = redis.from_url(url)

    @property
    def _messages_key(self) -> str:
        """Redis key for the messages list of the current conversation."""
        return f"{self._prefix}{self._conversation_id}:messages"

    @property
    def _conversations_key(self) -> str:
        """Redis key for the set of all conversation IDs."""
        return f"{self._prefix}conversations"

    def add(self, role: str, content: str, metadata: dict | None = None) -> None:
        """Append a message to the conversation."""
        msg: dict = {"role": role, "content": content}
        if metadata:
            msg["metadata"] = metadata
        self._client.rpush(self._messages_key, json.dumps(msg))
        self._client.sadd(self._conversations_key, self._conversation_id)

        # Apply window if set
        if self._window is not None:
            max_messages = self._window * 2
            self._client.ltrim(self._messages_key, -max_messages, -1)

    def get_messages(self) -> list[dict]:
        """Return all messages for this conversation."""
        raw = self._client.lrange(self._messages_key, 0, -1)
        return [json.loads(item) for item in raw]

    def format_context(self) -> str:
        """Flatten history to a plain string for prompt injection."""
        parts = []
        for m in self.get_messages():
            role = m["role"].capitalize()
            parts.append(f"{role}: {m['content']}")
        return "\n".join(parts)

    def clear(self) -> None:
        """Delete all messages for this conversation."""
        self._client.delete(self._messages_key)
        self._client.srem(self._conversations_key, self._conversation_id)

    def list_conversations(self) -> list[str]:
        """Return all conversation IDs tracked in Redis."""
        members = self._client.smembers(self._conversations_key)
        # smembers may return bytes or str depending on decode_responses
        return sorted(m.decode() if isinstance(m, bytes) else m for m in members)

    def __len__(self) -> int:
        return int(self._client.llen(self._messages_key))

    def close(self) -> None:
        """Close the Redis connection."""
        self._client.close()
