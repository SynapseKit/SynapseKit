"""Behavioral tests for AsyncLRUCache and ConversationMemory.

Verifies cache put/get/eviction/stats/make_key and ConversationMemory
window enforcement, add/get/clear, format_context, and edge cases.
"""

from __future__ import annotations

import pytest

from synapsekit.llm._cache import AsyncLRUCache
from synapsekit.memory.conversation import ConversationMemory

# ---------------------------------------------------------------------------
# AsyncLRUCache
# ---------------------------------------------------------------------------


class TestAsyncLRUCache:
    def test_put_and_get_hit(self):
        cache = AsyncLRUCache(maxsize=10)
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        assert cache.hits == 1
        assert cache.misses == 0

    def test_get_miss_returns_none(self):
        cache = AsyncLRUCache(maxsize=10)
        result = cache.get("nonexistent")
        assert result is None
        assert cache.misses == 1
        assert cache.hits == 0

    def test_lru_eviction_removes_oldest(self):
        cache = AsyncLRUCache(maxsize=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        # Access 'a' to make it recently used
        cache.get("a")
        # Insert 'd' — should evict 'b' (least recently used)
        cache.put("d", 4)
        assert cache.get("b") is None  # evicted
        assert cache.get("a") == 1  # still present
        assert cache.get("c") == 3  # still present
        assert cache.get("d") == 4  # just inserted

    def test_put_existing_key_moves_to_end(self):
        cache = AsyncLRUCache(maxsize=2)
        cache.put("a", 1)
        cache.put("b", 2)
        # Update 'a' — should move it to most recently used
        cache.put("a", 99)
        # Insert 'c' — should evict 'b', not 'a'
        cache.put("c", 3)
        assert cache.get("a") == 99
        assert cache.get("b") is None  # evicted
        assert cache.get("c") == 3

    def test_clear_empties_cache(self):
        cache = AsyncLRUCache(maxsize=5)
        cache.put("x", 1)
        cache.put("y", 2)
        cache.clear()
        assert len(cache) == 0
        assert cache.get("x") is None

    def test_len_reflects_size(self):
        cache = AsyncLRUCache(maxsize=10)
        assert len(cache) == 0
        cache.put("a", 1)
        assert len(cache) == 1
        cache.put("b", 2)
        assert len(cache) == 2

    def test_len_does_not_exceed_maxsize(self):
        cache = AsyncLRUCache(maxsize=3)
        for i in range(10):
            cache.put(f"key{i}", i)
        assert len(cache) <= 3

    def test_hit_miss_counters_accumulate(self):
        cache = AsyncLRUCache(maxsize=5)
        cache.put("k", "v")
        cache.get("k")  # hit
        cache.get("k")  # hit
        cache.get("miss1")  # miss
        cache.get("miss2")  # miss
        assert cache.hits == 2
        assert cache.misses == 2

    def test_make_key_deterministic(self):
        k1 = AsyncLRUCache.make_key("gpt-4o", "hello", 0.2, 1024)
        k2 = AsyncLRUCache.make_key("gpt-4o", "hello", 0.2, 1024)
        assert k1 == k2

    def test_make_key_differs_on_model(self):
        k1 = AsyncLRUCache.make_key("gpt-4o", "hello", 0.2, 1024)
        k2 = AsyncLRUCache.make_key("gpt-3.5", "hello", 0.2, 1024)
        assert k1 != k2

    def test_make_key_differs_on_prompt(self):
        k1 = AsyncLRUCache.make_key("gpt-4o", "hello", 0.2, 1024)
        k2 = AsyncLRUCache.make_key("gpt-4o", "world", 0.2, 1024)
        assert k1 != k2

    def test_make_key_differs_on_temperature(self):
        k1 = AsyncLRUCache.make_key("gpt-4o", "p", 0.0, 1024)
        k2 = AsyncLRUCache.make_key("gpt-4o", "p", 1.0, 1024)
        assert k1 != k2

    def test_make_key_with_messages_list(self):
        messages = [{"role": "user", "content": "hi"}]
        k1 = AsyncLRUCache.make_key("gpt-4o", messages, 0.2, 512)
        k2 = AsyncLRUCache.make_key("gpt-4o", messages, 0.2, 512)
        assert k1 == k2

    def test_cache_stores_any_type(self):
        cache = AsyncLRUCache(maxsize=5)
        cache.put("list", [1, 2, 3])
        cache.put("dict", {"a": 1})
        cache.put("none", None)
        # None stored is indistinguishable from miss — by design that's fine
        assert cache.get("list") == [1, 2, 3]
        assert cache.get("dict") == {"a": 1}


# ---------------------------------------------------------------------------
# ConversationMemory
# ---------------------------------------------------------------------------


class TestConversationMemory:
    def test_add_and_get_messages(self):
        mem = ConversationMemory(window=5)
        mem.add("user", "Hello")
        mem.add("assistant", "Hi there")
        msgs = mem.get_messages()
        assert len(msgs) == 2
        assert msgs[0] == {"role": "user", "content": "Hello"}
        assert msgs[1] == {"role": "assistant", "content": "Hi there"}

    def test_window_limits_messages(self):
        mem = ConversationMemory(window=2)
        # window=2 keeps 4 messages (2 turns x 2 messages per turn)
        for i in range(5):
            mem.add("user", f"q{i}")
            mem.add("assistant", f"a{i}")
        msgs = mem.get_messages()
        assert len(msgs) == 4  # last 2 turns only

    def test_sliding_window_drops_oldest(self):
        mem = ConversationMemory(window=1)
        mem.add("user", "first")
        mem.add("assistant", "first answer")
        mem.add("user", "second")
        mem.add("assistant", "second answer")
        msgs = mem.get_messages()
        assert len(msgs) == 2
        assert msgs[0]["content"] == "second"
        assert msgs[1]["content"] == "second answer"

    def test_clear_empties_history(self):
        mem = ConversationMemory(window=5)
        mem.add("user", "hello")
        mem.add("assistant", "hi")
        mem.clear()
        assert len(mem) == 0
        assert mem.get_messages() == []

    def test_len_reflects_message_count(self):
        mem = ConversationMemory(window=5)
        assert len(mem) == 0
        mem.add("user", "msg1")
        assert len(mem) == 1
        mem.add("assistant", "reply1")
        assert len(mem) == 2

    def test_format_context_flat_string(self):
        mem = ConversationMemory(window=5)
        mem.add("user", "What is 2+2?")
        mem.add("assistant", "It is 4.")
        ctx = mem.format_context()
        assert "User: What is 2+2?" in ctx
        assert "Assistant: It is 4." in ctx

    def test_format_context_empty(self):
        mem = ConversationMemory(window=5)
        assert mem.format_context() == ""

    def test_get_messages_returns_copy(self):
        """Modifying returned list does not affect internal state."""
        mem = ConversationMemory(window=5)
        mem.add("user", "hello")
        msgs = mem.get_messages()
        msgs.append({"role": "user", "content": "injected"})
        assert len(mem) == 1  # unchanged

    def test_window_1_is_valid(self):
        mem = ConversationMemory(window=1)
        mem.add("user", "only this")
        mem.add("assistant", "and this")
        assert len(mem.get_messages()) == 2

    def test_window_0_raises_value_error(self):
        with pytest.raises(ValueError, match="window must be >= 1"):
            ConversationMemory(window=0)

    def test_multiple_roles_preserved(self):
        mem = ConversationMemory(window=10)
        mem.add("system", "Be helpful")
        mem.add("user", "Hello")
        mem.add("assistant", "Hi!")
        msgs = mem.get_messages()
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"

    def test_format_context_capitalizes_role(self):
        mem = ConversationMemory(window=5)
        mem.add("user", "q")
        ctx = mem.format_context()
        assert ctx.startswith("User:")
