"""Tests for AgentMemory."""

from __future__ import annotations

from synapsekit.agents.memory import AgentMemory, AgentStep


class TestAgentMemory:
    def test_empty_on_init(self):
        mem = AgentMemory()
        assert len(mem) == 0
        assert mem.steps == []

    def test_add_step(self):
        mem = AgentMemory()
        step = AgentStep(
            thought="Let me calculate",
            action="calculator",
            action_input="2+2",
            observation="4",
        )
        mem.add_step(step)
        assert len(mem) == 1
        assert mem.steps[0].observation == "4"

    def test_multiple_steps(self):
        mem = AgentMemory()
        for i in range(3):
            mem.add_step(
                AgentStep(
                    thought=f"thought {i}",
                    action="calc",
                    action_input=str(i),
                    observation=str(i * 2),
                )
            )
        assert len(mem) == 3

    def test_format_scratchpad(self):
        mem = AgentMemory()
        mem.add_step(
            AgentStep(
                thought="I need to add",
                action="calculator",
                action_input="1+1",
                observation="2",
            )
        )
        scratchpad = mem.format_scratchpad()
        assert "Thought: I need to add" in scratchpad
        assert "Action: calculator" in scratchpad
        assert "Action Input: 1+1" in scratchpad
        assert "Observation: 2" in scratchpad

    def test_format_scratchpad_empty(self):
        mem = AgentMemory()
        assert mem.format_scratchpad() == ""

    def test_is_full(self):
        mem = AgentMemory(max_steps=2)
        assert not mem.is_full()
        mem.add_step(AgentStep("t", "a", "i", "o"))
        mem.add_step(AgentStep("t", "a", "i", "o"))
        assert mem.is_full()

    def test_clear(self):
        mem = AgentMemory()
        mem.add_step(AgentStep("t", "a", "i", "o"))
        mem.clear()
        assert len(mem) == 0

    def test_steps_returns_copy(self):
        mem = AgentMemory()
        mem.add_step(AgentStep("t", "a", "i", "o"))
        steps = mem.steps
        steps.clear()
        assert len(mem) == 1  # original unaffected
