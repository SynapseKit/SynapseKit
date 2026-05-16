import pytest

from synapsekit.memory.smart_context import SmartContextManager


class MockLLM:
    """Mock LLM to track prompt passing and return a fixed summary."""

    def __init__(self, summary: str = "Mock summary"):
        self.summary = summary
        self.call_count = 0
        self.last_prompt = ""

    async def generate(self, prompt: str) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        return self.summary


@pytest.mark.asyncio
async def test_smart_context_manager_basic():
    manager = SmartContextManager(max_recent_tokens=100, chars_per_token=4)
    manager.set_system("You are a smart assistant.")
    manager.add("user", "Hello there!")
    manager.add("assistant", "Hi, how can I help?")

    messages = await manager.get_messages()

    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a smart assistant."
    assert messages[0].get("cache_control") == {"type": "ephemeral"}
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"


@pytest.mark.asyncio
async def test_smart_context_manager_summarization():
    mock_llm = MockLLM(summary="A very short summary of a long chat.")
    # Tiny limit to force summarization early
    manager = SmartContextManager(cheap_llm=mock_llm, max_recent_tokens=10, chars_per_token=1)

    manager.set_system("Sys prompt")
    manager.add("user", "This is message 1. " * 5)
    manager.add("assistant", "This is reply 1. " * 5)
    manager.add("user", "This is message 2. " * 5)
    manager.add("assistant", "This is reply 2. " * 5)

    messages = await manager.get_messages()

    assert mock_llm.call_count > 0
    # Expected messages: system prompt, summary, and remaining recent messages
    roles = [m["role"] for m in messages]
    assert roles.count("system") == 2  # 1 for system, 1 for summary

    summary_message = next(m for m in messages if "Summary of earlier" in m["content"])
    assert summary_message["cache_control"] == {"type": "ephemeral"}
    assert "A very short summary" in summary_message["content"]


@pytest.mark.asyncio
async def test_smart_context_manager_search_bounding():
    manager = SmartContextManager(max_recent_tokens=100, max_search_tokens=10, chars_per_token=1)
    long_search_result = "A" * 100
    manager.set_search_results(long_search_result)

    manager.add("user", "Question")

    messages = await manager.get_messages()

    search_message = next(m for m in messages if "Search Results Context:" in m["content"])
    # 10 tokens * 1 char/token = 10 chars + prefix length
    assert len(search_message["content"]) == len("Search Results Context:\n") + 10


@pytest.mark.asyncio
async def test_smart_context_manager_large_conversation():
    """Simulate a >100K token conversation to verify pruning logic handles scale."""
    mock_llm = MockLLM(summary="Compressed summary of 100K tokens.")
    manager = SmartContextManager(cheap_llm=mock_llm, max_recent_tokens=4000, chars_per_token=4)

    manager.set_system("You are handling a huge document.")

    # Generate ~100K tokens. 100,000 * 4 chars = 400,000 chars.
    # 100 messages of 4,000 chars each.
    for i in range(100):
        manager.add("user", f"Question {i} " + "A" * 3900)
        manager.add("assistant", f"Answer {i} " + "B" * 3900)

    # Calling get_messages should trigger summarization until we're under the max_recent_tokens budget
    messages = await manager.get_messages()

    # Budget is 4000 tokens (16,000 chars).
    # Each exchange (user + assistant) is ~8000 chars = ~2000 tokens.
    # So we should retain roughly 2 exchanges (4 messages) in the recent buffer, plus system & summary.

    assert mock_llm.call_count > 0
    assert len(messages) <= 10  # system, summary, plus a few recent messages

    roles = [m["role"] for m in messages]
    assert roles.count("system") == 2

    summary_message = next(m for m in messages if "Summary of earlier" in m["content"])
    assert "Compressed summary" in summary_message["content"]
