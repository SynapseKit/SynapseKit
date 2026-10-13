import os

import pytest

from synapsekit.llm.base import LLMConfig
from synapsekit.llm.openai import OpenAILLM
from synapsekit.testing.vcr import use_cassette

# This test requires an OpenAI API key on the first run to record the cassette.
# Once the cassette is recorded, you can run this test even without internet or an API key.
# No cassette is checked into the repo, so CI (which has no OpenAI credentials for
# fork PRs) skips on a fresh checkout rather than making a live network call per
# CONTRIBUTING.md's "no test should require an API key or network access".
_CASSETTE_PATH = os.path.join(os.path.dirname(__file__), "cassettes", "openai_demo.yaml")


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY") and not os.path.exists(_CASSETTE_PATH),
    reason="requires OPENAI_API_KEY to record the cassette (or an existing one to replay)",
)
async def test_live_openai_with_vcr():
    # Falls back to a dummy key when replaying from an already-recorded cassette.
    api_key = os.environ.get("OPENAI_API_KEY", "sk-fake-key-for-cached-runs")

    config = LLMConfig(
        model="gpt-4o-mini",
        api_key=api_key,
        provider="openai",
        system_prompt="You are a helpful assistant.",
        temperature=0.0,
    )
    llm = OpenAILLM(config)

    with use_cassette(_CASSETTE_PATH):
        result = await llm.generate("What is the capital of France? Reply in one word.")
        assert "Paris" in result

    print("\n✅ VCR Test Passed! Result:", result)
    print("Cassette stored at:", _CASSETTE_PATH)
