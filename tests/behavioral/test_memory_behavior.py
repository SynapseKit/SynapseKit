"""Behavioral tests — ConversationMemory and AgentMemory.

Verifies correctness of memory semantics:
  - Window enforcement (old messages evicted)
  - Persistence across multiple calls
  - Clear resets state
  - AgentMemory step tracking
"""

from __future__ import annotations

import pytest

from synapsekit.memory.conversation import ConversationMemory

# ---------------------------------------------------------------------------
# ConversationMemory
# ---------------------------------------------------------------------------


class TestConversationMemoryBasics:
    def test_starts_empty(self):
        mem = ConversationMemory(window=5)
        assert mem.get_messages() == []
        assert len(mem) == 0

    def test_add_message(self):
        mem = ConversationMemory(window=5)
        mem.add("user", "Hello")
        msgs = mem.get_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Hello"

    def test_add_multiple_messages(self):
        mem = ConversationMemory(window=5)
        mem.add("user", "Q1")
        mem.add("assistant", "A1")
        mem.add("user", "Q2")
        assert len(mem) == 3

    def test_clear_resets(self):
        mem = ConversationMemory(window=5)
        mem.add("user", "question")
        mem.add("assistant", "answer")
        mem.clear()
        assert mem.get_messages() == []

    def test_clear_then_add(self):
        mem = ConversationMemory(window=5)
        mem.add("user", "first")
        mem.clear()
        mem.add("user", "second")
        assert len(mem) == 1
        assert mem.get_messages()[0]["content"] == "second"


class TestConversationMemoryWindow:
    def test_window_evicts_oldest_messages(self):
        """window=1 keeps 2 messages max (1 user + 1 assistant pair)."""
        mem = ConversationMemory(window=1)
        mem.add("user", "msg1")
        mem.add("assistant", "resp1")
        mem.add("user", "msg2")
        # After 3 inserts with window=1 (max=2 messages), only last 2 should remain
        assert len(mem) <= 2
        contents = [m["content"] for m in mem.get_messages()]
        assert "msg2" in contents

    def test_window_2_keeps_four_messages(self):
        """window=2 keeps at most 4 messages (2 user+assistant pairs)."""
        mem = ConversationMemory(window=2)
        for i in range(6):
            mem.add("user", f"Q{i}")
            mem.add("assistant", f"A{i}")
        assert len(mem) <= 4

    def test_window_most_recent_always_present(self):
        mem = ConversationMemory(window=1)
        for i in range(10):
            mem.add("user", f"msg{i}")
        msgs = mem.get_messages()
        contents = [m["content"] for m in msgs]
        assert "msg9" in contents

    def test_window_large_keeps_all(self):
        """Large window keeps all messages."""
        mem = ConversationMemory(window=100)
        for i in range(20):
            mem.add("user", f"msg{i}")
        assert len(mem) == 20

    def test_window_below_one_raises(self):
        with pytest.raises(ValueError, match="window"):
            ConversationMemory(window=0)


class TestConversationMemoryFormat:
    def test_messages_are_dicts_with_role_content(self):
        mem = ConversationMemory(window=10)
        mem.add("user", "test message")
        msg = mem.get_messages()[0]
        assert isinstance(msg, dict)
        assert "role" in msg
        assert "content" in msg

    def test_system_role_accepted(self):
        mem = ConversationMemory(window=10)
        mem.add("system", "You are helpful.")
        assert mem.get_messages()[0]["role"] == "system"

    def test_message_content_preserved_exactly(self):
        mem = ConversationMemory(window=10)
        content = "Special chars: !@#$%^&*() Unicode: 🚀 Newlines:\nhere"
        mem.add("user", content)
        assert mem.get_messages()[0]["content"] == content

    def test_format_context_non_empty(self):
        mem = ConversationMemory(window=5)
        mem.add("user", "Hello")
        mem.add("assistant", "Hi there")
        ctx = mem.format_context()
        assert "Hello" in ctx
        assert "Hi there" in ctx


# ---------------------------------------------------------------------------
# AgentMemory
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestAgentMemory:
    def test_starts_empty(self):
        from synapsekit.agents.memory import AgentMemory

        mem = AgentMemory(max_steps=5)
        assert mem.steps == []
        assert len(mem) == 0

    def test_add_step(self):
        from synapsekit.agents.memory import AgentMemory, AgentStep

        mem = AgentMemory(max_steps=5)
        step = AgentStep(
            thought="I need to calculate",
            action="calculator",
            action_input="2+2",  # action_input is str, not dict
            observation="4",
        )
        mem.add_step(step)
        assert len(mem) == 1

    def test_is_full_false_when_not_at_limit(self):
        from synapsekit.agents.memory import AgentMemory, AgentStep

        mem = AgentMemory(max_steps=3)
        step = AgentStep(thought="t", action="a", action_input="x", observation="o")
        mem.add_step(step)
        assert not mem.is_full()

    def test_is_full_true_at_limit(self):
        from synapsekit.agents.memory import AgentMemory, AgentStep

        mem = AgentMemory(max_steps=2)
        for _ in range(2):
            step = AgentStep(thought="t", action="a", action_input="x", observation="o")
            mem.add_step(step)
        assert mem.is_full()

    def test_clear_resets_steps(self):
        from synapsekit.agents.memory import AgentMemory, AgentStep

        mem = AgentMemory(max_steps=5)
        step = AgentStep(thought="t", action="a", action_input="x", observation="o")
        mem.add_step(step)
        mem.clear()
        assert mem.steps == []

    def test_step_fields_preserved(self):
        from synapsekit.agents.memory import AgentMemory, AgentStep

        mem = AgentMemory(max_steps=10)
        step = AgentStep(
            thought="calculate",
            action="calculator",
            action_input="1+1",
            observation="2",
        )
        mem.add_step(step)
        stored = mem.steps[0]
        assert stored.thought == "calculate"
        assert stored.action == "calculator"
        assert stored.observation == "2"

    def test_format_scratchpad_contains_steps(self):
        from synapsekit.agents.memory import AgentMemory, AgentStep

        mem = AgentMemory(max_steps=10)
        mem.add_step(AgentStep(thought="think", action="tool", action_input="x", observation="y"))
        scratchpad = mem.format_scratchpad()
        assert "think" in scratchpad
        assert "tool" in scratchpad
        assert "y" in scratchpad
