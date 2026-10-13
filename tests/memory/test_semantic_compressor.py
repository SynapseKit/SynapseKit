import pytest

from synapsekit.memory.semantic_compressor import SemanticCompressorMemory


class FakeCompressorLLM:
    """Hand-written fake standing in for a BaseLLM compressor."""

    def __init__(
        self, response: str = "Compressed summary: User wants a feature and likes apples."
    ):
        self._response = response
        self.call_count = 0

    async def generate(self, prompt: str) -> str:
        self.call_count += 1
        return self._response


@pytest.fixture
def fake_llm():
    return FakeCompressorLLM()


@pytest.mark.asyncio
async def test_semantic_compressor_basic(fake_llm):
    memory = SemanticCompressorMemory(
        llm=fake_llm,
        max_tokens=1000,
        compression_threshold=0.8,  # Threshold = 800 tokens
        chars_per_token=4,
    )

    assert len(memory) == 0

    # 800 tokens = 3200 characters
    # We will add 5 messages, each ~700 characters
    long_string = "a" * 700
    for i in range(5):
        memory.add("user", f"Message {i}: {long_string}")

    # Total tokens = 5 * (700 // 4) = 5 * 175 = 875 tokens > 800 threshold.
    # It should trigger compression.
    messages = await memory.get_messages()

    # Half of 5 is 2. So 2 messages are compressed, 3 remain.
    # But then 1 system summary message is added at the start.
    assert len(messages) == 4

    assert messages[0]["role"] == "system"
    assert "Compressed summary" in messages[0]["content"]

    assert messages[1]["content"].startswith("Message 2")
    assert messages[2]["content"].startswith("Message 3")
    assert messages[3]["content"].startswith("Message 4")

    assert fake_llm.call_count == 1

    # Format context test
    ctx = memory.format_context()
    assert "Compressed Memory:\nCompressed summary" in ctx
    assert "Message 2" in ctx

    # Clear
    memory.clear()
    assert len(memory) == 0
    assert memory.summary == ""


def test_semantic_compressor_invalid_tokens():
    with pytest.raises(ValueError):
        SemanticCompressorMemory(llm=FakeCompressorLLM(), max_tokens=100)
