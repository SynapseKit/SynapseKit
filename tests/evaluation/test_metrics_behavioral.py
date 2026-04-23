"""Behavioral tests for evaluation metrics.

Tests that FaithfulnessMetric, GroundednessMetric, and RelevancyMetric
return correct scores and MetricResult fields under all branching conditions.
No real LLM calls — all llm.generate() calls are mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from synapsekit.evaluation.faithfulness import FaithfulnessMetric
from synapsekit.evaluation.groundedness import GroundednessMetric
from synapsekit.evaluation.relevancy import RelevancyMetric


def _mock_llm(*responses: str):
    """Return a mock LLM whose generate() cycles through *responses* in order."""
    llm = AsyncMock()
    llm.generate = AsyncMock(side_effect=list(responses))
    return llm


# ---------------------------------------------------------------------------
# FaithfulnessMetric
# ---------------------------------------------------------------------------


class TestFaithfulnessMetric:
    @pytest.mark.asyncio
    async def test_all_claims_supported_returns_1(self):
        llm = _mock_llm(
            "1. Python is a language.\n2. Guido created it.",  # claims
            "YES",  # claim 1 supported
            "YES",  # claim 2 supported
        )
        metric = FaithfulnessMetric(llm)
        result = await metric.evaluate(
            question="What is Python?",
            answer="Python is a language. Guido created it.",
            contexts=["Python is a programming language created by Guido."],
        )
        assert result.score == pytest.approx(1.0)
        assert "2/2" in result.reason

    @pytest.mark.asyncio
    async def test_partial_support_returns_fractional_score(self):
        llm = _mock_llm(
            "1. Python is a language.\n2. Python was created in 1980.",  # claims
            "YES",  # claim 1 supported
            "NO",  # claim 2 not supported
        )
        metric = FaithfulnessMetric(llm)
        result = await metric.evaluate(
            question="Tell me about Python",
            answer="Python is a language. It was created in 1980.",
            contexts=["Python is a language created in 1991."],
        )
        assert result.score == pytest.approx(0.5)
        assert result.details["claims"] is not None
        assert len(result.details["supported"]) == 2

    @pytest.mark.asyncio
    async def test_no_claims_none_response_returns_1(self):
        """If LLM says NONE, score=1.0."""
        llm = _mock_llm("NONE")
        metric = FaithfulnessMetric(llm)
        result = await metric.evaluate(question="q", answer="I don't know.", contexts=["context"])
        assert result.score == pytest.approx(1.0)
        assert result.details["claims"] == []

    @pytest.mark.asyncio
    async def test_empty_claims_list_returns_1(self):
        """If LLM returns non-numbered text, score=1.0."""
        llm = _mock_llm("No claims here")
        metric = FaithfulnessMetric(llm)
        result = await metric.evaluate(question="q", answer="answer", contexts=["ctx"])
        assert result.score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_all_claims_unsupported_returns_0(self):
        llm = _mock_llm(
            "1. Claim A\n2. Claim B",
            "NO",
            "NO",
        )
        metric = FaithfulnessMetric(llm)
        result = await metric.evaluate(
            question="q", answer="Claim A. Claim B.", contexts=["unrelated ctx"]
        )
        assert result.score == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_result_has_correct_fields(self):
        llm = _mock_llm("1. Fact one.", "YES")
        metric = FaithfulnessMetric(llm)
        result = await metric.evaluate(
            question="q", answer="Fact one.", contexts=["Fact one is true."]
        )
        assert hasattr(result, "score")
        assert hasattr(result, "reason")
        assert hasattr(result, "details")
        assert "claims" in result.details
        assert "supported" in result.details

    @pytest.mark.asyncio
    async def test_multiple_contexts_joined(self):
        """Multiple contexts are concatenated properly."""
        captured = []
        llm = AsyncMock()

        async def _generate(prompt):
            captured.append(prompt)
            if len(captured) == 1:
                return "1. Claim from multi-context."
            return "YES"

        llm.generate = _generate
        metric = FaithfulnessMetric(llm)
        await metric.evaluate(
            question="q",
            answer="Claim from multi-context.",
            contexts=["ctx A", "ctx B"],
        )
        # The verification prompt should contain both sources
        assert "[Source 1]" in captured[1]
        assert "[Source 2]" in captured[1]


# ---------------------------------------------------------------------------
# GroundednessMetric
# ---------------------------------------------------------------------------


class TestGroundednessMetric:
    @pytest.mark.asyncio
    async def test_empty_answer_returns_1(self):
        llm = _mock_llm()
        metric = GroundednessMetric(llm)
        result = await metric.evaluate(answer="", contexts=["some context"])
        assert result.score == pytest.approx(1.0)
        assert "Empty answer" in result.reason

    @pytest.mark.asyncio
    async def test_no_contexts_returns_0(self):
        llm = _mock_llm()
        metric = GroundednessMetric(llm)
        result = await metric.evaluate(answer="some answer", contexts=[])
        assert result.score == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_perfect_score_returns_1(self):
        llm = _mock_llm("10")  # LLM rates 10/10
        metric = GroundednessMetric(llm)
        result = await metric.evaluate(
            answer="Python was created by Guido.",
            contexts=["Python was created by Guido van Rossum."],
        )
        assert result.score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_zero_score_from_llm(self):
        llm = _mock_llm("0")
        metric = GroundednessMetric(llm)
        result = await metric.evaluate(
            answer="Python was created on the moon.",
            contexts=["Python was created by Guido."],
        )
        assert result.score == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_mid_score_normalized(self):
        llm = _mock_llm("5")
        metric = GroundednessMetric(llm)
        result = await metric.evaluate(answer="partial answer", contexts=["context"])
        assert result.score == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_unparseable_response_defaults_to_half(self):
        llm = _mock_llm("I cannot determine this.")
        metric = GroundednessMetric(llm)
        result = await metric.evaluate(answer="answer", contexts=["ctx"])
        assert result.score == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_score_clamped_to_0_1(self):
        """Scores outside [0, 10] are clamped to [0, 1]."""
        llm = _mock_llm("15")
        metric = GroundednessMetric(llm)
        result = await metric.evaluate(answer="answer", contexts=["ctx"])
        assert result.score <= 1.0

    @pytest.mark.asyncio
    async def test_raw_response_in_details(self):
        llm = _mock_llm("7")
        metric = GroundednessMetric(llm)
        result = await metric.evaluate(answer="answer", contexts=["ctx"])
        assert "raw_response" in result.details
        assert result.details["raw_response"] == "7"


# ---------------------------------------------------------------------------
# RelevancyMetric
# ---------------------------------------------------------------------------


class TestRelevancyMetric:
    @pytest.mark.asyncio
    async def test_no_contexts_returns_0(self):
        llm = _mock_llm()
        metric = RelevancyMetric(llm)
        result = await metric.evaluate(question="q", contexts=[])
        assert result.score == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_all_relevant_returns_1(self):
        llm = _mock_llm("YES", "YES", "YES")
        metric = RelevancyMetric(llm)
        result = await metric.evaluate(
            question="What is Python?",
            contexts=["ctx1", "ctx2", "ctx3"],
        )
        assert result.score == pytest.approx(1.0)
        assert "3/3" in result.reason

    @pytest.mark.asyncio
    async def test_none_relevant_returns_0(self):
        llm = _mock_llm("NO", "NO")
        metric = RelevancyMetric(llm)
        result = await metric.evaluate(question="What is Python?", contexts=["weather", "sports"])
        assert result.score == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_mixed_relevancy_returns_fraction(self):
        llm = _mock_llm("YES", "NO", "YES", "NO")
        metric = RelevancyMetric(llm)
        result = await metric.evaluate(question="Python?", contexts=["c1", "c2", "c3", "c4"])
        assert result.score == pytest.approx(0.5)
        assert "2/4" in result.reason

    @pytest.mark.asyncio
    async def test_relevancy_scores_in_details(self):
        llm = _mock_llm("YES", "NO")
        metric = RelevancyMetric(llm)
        result = await metric.evaluate(question="q", contexts=["c1", "c2"])
        assert result.details["relevancy_scores"] == [True, False]

    @pytest.mark.asyncio
    async def test_case_insensitive_yes(self):
        """YES in various cases should be recognized."""
        llm = _mock_llm("yes", "Yes", "YES")
        metric = RelevancyMetric(llm)
        result = await metric.evaluate(question="q", contexts=["c1", "c2", "c3"])
        assert result.score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_each_context_gets_own_prompt(self):
        """Each context gets an individual LLM call."""
        call_count = 0
        llm = AsyncMock()

        async def _generate(prompt):
            nonlocal call_count
            call_count += 1
            return "YES"

        llm.generate = _generate
        metric = RelevancyMetric(llm)
        await metric.evaluate(question="q", contexts=["c1", "c2", "c3", "c4"])
        assert call_count == 4
