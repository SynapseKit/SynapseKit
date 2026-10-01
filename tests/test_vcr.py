import asyncio
import os
import shutil
from typing import Any, AsyncGenerator

import pytest

from synapsekit.llm.base import BaseLLM, LLMConfig
from synapsekit.testing.vcr import use_cassette

class MockLLM(BaseLLM):
    """A mock LLM for testing VCR."""
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.call_count = 0

    async def stream(self, prompt: str, **kw: Any) -> AsyncGenerator[str, None]:
        self.call_count += 1
        yield f"Mock response for: {prompt}"

@pytest.fixture
def temp_cassette_dir(tmp_path):
    yield tmp_path / "cassettes"

@pytest.mark.asyncio
async def test_vcr_record_and_replay(temp_cassette_dir):
    cassette_path = str(temp_cassette_dir / "test.yaml")
    config = LLMConfig(model="test-model", api_key="test", provider="test")
    llm = MockLLM(config)

    # 1. Record
    with use_cassette(cassette_path):
        result = await llm.generate("Hello!")
        assert result == "Mock response for: Hello!"
        assert llm.call_count == 1

    # 2. Replay
    with use_cassette(cassette_path):
        result = await llm.generate("Hello!")
        assert result == "Mock response for: Hello!"
        assert llm.call_count == 1  # Should NOT increment

    # 3. Different prompt should miss cache
    with use_cassette(cassette_path):
        result = await llm.generate("Different prompt")
        assert result == "Mock response for: Different prompt"
        assert llm.call_count == 2
