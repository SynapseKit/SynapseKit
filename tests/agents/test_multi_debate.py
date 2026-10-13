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


class _RecordingLLM(BaseLLM):
    """Fake that returns scripted responses and records every prompt it receives.

    ReActAgent clears its scratchpad on every run(), so the *only* way earlier
    debate context can reach this LLM is if the orchestrator re-injects it into
    the prompt text — this is what the regression test below checks for.
    """

    def __init__(self, responses: list[str]):
        super().__init__(LLMConfig(model="mock", api_key="test", provider="openai"))
        self._responses = list(responses)
        self.seen_prompts: list[str] = []

    async def stream(self, prompt: str, **kw):
        yield self._responses[0]

    async def generate_with_messages(self, messages: list[dict], **kw) -> str:
        self.seen_prompts.append(messages[1]["content"])
        return self._responses.pop(0)


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


@pytest.mark.asyncio
async def test_debate_orchestrator_does_not_forget_earlier_rounds() -> None:
    """Regression test: round-2 prompts must carry the full transcript.

    Each executor's ReActAgent wipes its own scratchpad between run() calls,
    so without re-injecting the transcript, round 2 would only ever see the
    single latest message and "forget" what was argued in round 1.
    """
    proposer_llm = _RecordingLLM(
        ["Opening argument.", "Proposer defense R1", "Proposer defense R2"]
    )
    opponent_llm = _RecordingLLM(["Opponent rebuttal R1", "Opponent rebuttal R2"])
    judge_llm = _MockLLM(response="Final judged decision.")

    debate = DebateOrchestrator(
        topic="Microservices vs Monolith",
        proposer=Debater(name="Alice", role="Microservices Advocate", llm=proposer_llm),
        opponent=Debater(name="Bob", role="Monolith Advocate", llm=opponent_llm),
        judge=Judge(name="Charlie", role="Senior Architect", llm=judge_llm),
        rounds=2,
    )

    await debate.run()

    # Opponent's round-2 rebuttal prompt must still reference round-1 content,
    # not just the proposer's single latest message.
    round_2_rebuttal_prompt = opponent_llm.seen_prompts[1]
    assert "Opening argument." in round_2_rebuttal_prompt
    assert "Opponent rebuttal R1" in round_2_rebuttal_prompt
    assert "Proposer defense R1" in round_2_rebuttal_prompt

    # Proposer's round-2 defense prompt must still reference round-1 content.
    round_2_defense_prompt = proposer_llm.seen_prompts[2]
    assert "Opening argument." in round_2_defense_prompt
    assert "Opponent rebuttal R1" in round_2_defense_prompt
