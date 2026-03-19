"""Tests for RedisConversationMemory."""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

import pytest

# The redis module is lazily imported inside __init__, so we need to
# ensure a mock redis module is available *before* the first instantiation.
# We install a persistent mock in sys.modules so the class can be imported
# at module level and every test can create instances freely.
_mock_redis_module = MagicMock()
_original_redis = sys.modules.get("redis", None)
sys.modules["redis"] = _mock_redis_module

from synapsekit.memory.redis import RedisConversationMemory  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_redis_client():
    """Give each test a fresh MagicMock as the Redis client."""
    mock_client = MagicMock()
    mock_client.lrange.return_value = []
    mock_client.llen.return_value = 0
    mock_client.smembers.return_value = set()
    _mock_redis_module.from_url.return_value = mock_client
    yield mock_client


class TestRedisConversationMemory:
    def test_add_message(self, _fresh_redis_client):
        client = _fresh_redis_client
        mem = RedisConversationMemory(conversation_id="c1")
        mem.add("user", "Hello")

        expected_key = "synapsekit:memory:c1:messages"
        expected_json = json.dumps({"role": "user", "content": "Hello"})
        client.rpush.assert_called_once_with(expected_key, expected_json)
        client.sadd.assert_called_once_with(
            "synapsekit:memory:conversations", "c1"
        )

    def test_add_with_window(self, _fresh_redis_client):
        client = _fresh_redis_client
        mem = RedisConversationMemory(conversation_id="c1", window=3)
        mem.add("user", "Hello")

        # window=3 -> max 6 messages, ltrim keeps last 6
        client.ltrim.assert_called_once_with(
            "synapsekit:memory:c1:messages", -6, -1
        )

    def test_get_messages(self, _fresh_redis_client):
        client = _fresh_redis_client
        client.lrange.return_value = [
            json.dumps({"role": "user", "content": "Hi"}),
            json.dumps({"role": "assistant", "content": "Hello!"}),
        ]
        mem = RedisConversationMemory(conversation_id="c1")
        messages = mem.get_messages()

        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "Hi"}
        assert messages[1] == {"role": "assistant", "content": "Hello!"}
        client.lrange.assert_called_once_with(
            "synapsekit:memory:c1:messages", 0, -1
        )

    def test_format_context(self, _fresh_redis_client):
        client = _fresh_redis_client
        client.lrange.return_value = [
            json.dumps({"role": "user", "content": "What is Python?"}),
            json.dumps({"role": "assistant", "content": "A programming language."}),
        ]
        mem = RedisConversationMemory(conversation_id="c1")
        ctx = mem.format_context()

        assert "User: What is Python?" in ctx
        assert "Assistant: A programming language." in ctx

    def test_clear(self, _fresh_redis_client):
        client = _fresh_redis_client
        mem = RedisConversationMemory(conversation_id="c1")
        mem.clear()

        client.delete.assert_called_once_with("synapsekit:memory:c1:messages")
        client.srem.assert_called_once_with(
            "synapsekit:memory:conversations", "c1"
        )

    def test_list_conversations(self, _fresh_redis_client):
        client = _fresh_redis_client
        client.smembers.return_value = {"beta", "alpha", "gamma"}
        mem = RedisConversationMemory()
        convos = mem.list_conversations()

        assert convos == ["alpha", "beta", "gamma"]

    def test_len(self, _fresh_redis_client):
        client = _fresh_redis_client
        client.llen.return_value = 5
        mem = RedisConversationMemory(conversation_id="c1")

        assert len(mem) == 5
        client.llen.assert_called_with("synapsekit:memory:c1:messages")

    def test_add_with_metadata(self, _fresh_redis_client):
        client = _fresh_redis_client
        mem = RedisConversationMemory(conversation_id="c1")
        mem.add("user", "Hello", metadata={"source": "web"})

        expected_json = json.dumps(
            {"role": "user", "content": "Hello", "metadata": {"source": "web"}}
        )
        client.rpush.assert_called_once_with(
            "synapsekit:memory:c1:messages", expected_json
        )

    def test_custom_prefix(self, _fresh_redis_client):
        client = _fresh_redis_client
        mem = RedisConversationMemory(
            conversation_id="c1", prefix="myapp:chat:"
        )
        mem.add("user", "test")

        expected_key = "myapp:chat:c1:messages"
        client.rpush.assert_called_once_with(
            expected_key, json.dumps({"role": "user", "content": "test"})
        )
        client.sadd.assert_called_once_with("myapp:chat:conversations", "c1")

    def test_import_error(self):
        """Verify a clear error when redis is not installed."""
        # Temporarily remove redis from sys.modules and make import fail
        saved = sys.modules.pop("redis", None)
        sys.modules["redis"] = None  # triggers ImportError on `import redis`
        try:
            # We need to force re-execution of `import redis` inside __init__,
            # so we create a fresh class by re-reading the source module.
            import importlib

            import synapsekit.memory.redis as redis_mod

            importlib.reload(redis_mod)
            with pytest.raises(ImportError, match="redis package required"):
                redis_mod.RedisConversationMemory()
        finally:
            # Restore the mock so other tests are unaffected
            sys.modules["redis"] = saved if saved is not None else _mock_redis_module
