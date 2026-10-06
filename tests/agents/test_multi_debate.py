"""Tests for DebateOrchestrator."""

from __future__ import annotations

import pytest

from synapsekit.agents.multi.debate import DebateOrchestrator, Debater, Judge
from synapsekit.llm.base import BaseLLM, LLMConfig


class _MockLLM(BaseLLM):
    def __init__(self, response: str = "Mock response"):
        super().__init__(LLMConfig(model="mock", api_key="test", provider="openai"))
        self._response = response

    async def stream(self, prompt: str, **kw):
        yield self._response

    async def generate(self, prompt: str, **kw):
        return self._response


@pytest.mark.asyncio
async def test_debate_orchestrator() -> None:
    """Test DebateOrchestrator execution."""
    proposer_llm = _MockLLM(response="Proposer argument.")
    opponent_llm = _MockLLM(response="Opponent rebuttal.")
    judge_llm = _MockLLM(response="Final judged decision.")

    debate = DebateOrchestrator(
        topic="Microservices vs Monolith",
        proposer=Debater(name="Alice", role="Microservices Advocate", llm=proposer_llm),
        opponent=Debater(name="Bob", role="Monolith Advocate", llm=opponent_llm),
        judge=Judge(name="Charlie", role="Senior Architect", llm=judge_llm),
        rounds=2,
    )

    result = await debate.run()

    # The judge's final decision should be output
    assert result.final_decision == "Final judged decision."

    # Transcript should contain:
    # Round 1: Proposer
    # Round 1: Opponent
    # Round 1: Proposer (defends)
    # Round 2: Opponent
    # Round 2: Proposer
    # Total transcript items: 1 (initial) + 2 rounds * 2 (rebuttal + defense) = 5
    assert len(result.transcript) == 5
    assert "Alice (Proposer):" in result.transcript[0]
    assert "Bob (Opponent):" in result.transcript[1]
