"""Tests for ConversationMemory."""
from __future__ import annotations

import pytest

from synapsekit.memory.conversation import ConversationMemory


class TestConversationMemory:
    def test_add_and_get_messages(self):
        mem = ConversationMemory(window=5)
        mem.add("user", "Hello")
        mem.add("assistant", "Hi there")
        messages = mem.get_messages()
        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "content": "Hi there"}

    def test_sliding_window_trims_oldest(self):
        mem = ConversationMemory(window=2)
        for i in range(3):
            mem.add("user", f"msg {i}")
            mem.add("assistant", f"reply {i}")

        # window=2 means max 4 messages
        assert len(mem) == 4
        messages = mem.get_messages()
        assert messages[0]["content"] == "msg 1"
        assert messages[-1]["content"] == "reply 2"

    def test_clear(self):
        mem = ConversationMemory()
        mem.add("user", "Hello")
        mem.clear()
        assert len(mem) == 0
        assert mem.get_messages() == []

    def test_format_context(self):
        mem = ConversationMemory()
        mem.add("user", "What is Python?")
        mem.add("assistant", "A programming language.")
        ctx = mem.format_context()
        assert "User: What is Python?" in ctx
        assert "Assistant: A programming language." in ctx

    def test_format_context_empty(self):
        mem = ConversationMemory()
        assert mem.format_context() == ""

    def test_get_messages_returns_copy(self):
        mem = ConversationMemory()
        mem.add("user", "test")
        msgs = mem.get_messages()
        msgs.append({"role": "user", "content": "injected"})
        assert len(mem) == 1  # original unaffected
