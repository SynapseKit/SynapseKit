import os
import pytest
from synapsekit.llm.base import LLMConfig
from synapsekit.llm.openai import OpenAILLM
from synapsekit.testing.vcr import use_cassette

# This test requires an OpenAI API key on the first run to record the cassette.
# Once the cassette is recorded, you can run this test even without internet or an API key!
@pytest.mark.asyncio
async def test_live_openai_with_vcr(tmp_path):
    # Set a dummy key if running from cache, or use the real one to record
    api_key = os.environ.get("OPENAI_API_KEY", "sk-fake-key-for-cached-runs")
    
    config = LLMConfig(
        model="gpt-4o-mini",
        api_key=api_key,
        provider="openai",
        system_prompt="You are a helpful assistant.",
        temperature=0.0
    )
    llm = OpenAILLM(config)
    
    # We will save the recorded LLM response here
    cassette_path = os.path.join(os.path.dirname(__file__), "cassettes", "openai_demo.yaml")
    
    with use_cassette(cassette_path):
        result = await llm.generate("What is the capital of France? Reply in one word.")
        assert "Paris" in result

    print("\n✅ VCR Test Passed! Result:", result)
    print("Cassette stored at:", cassette_path)
