from __future__ import annotations

import inspect
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest

from synapsekit.llm.base import BaseLLM, LLMConfig
from synapsekit.twin import (
    ApprovalRequiredError,
    AutoSendForbiddenError,
    DelegationPolicy,
    DigitalTwinAgent,
    DraftResult,
    LearnedPatterns,
    StyleProfile,
    VoiceMatcher,
    VoiceMatchResult,
)


class FakeLLM(BaseLLM):
    def __init__(self, response: str = "LGTM! Looks good to me.") -> None:
        super().__init__(
            LLMConfig(
                model="fake",
                api_key="",
                provider="fake",
            )
        )
        self.response = response
        self.prompts: list[str] = []

    async def stream(self, prompt: str, **kw: Any) -> AsyncGenerator[str, None]:
        self.prompts.append(prompt)
        yield self.response


def test_style_profile_load_default(tmp_path: Path) -> None:
    profile_file = tmp_path / "style.md"
    profile = StyleProfile(str(profile_file))
    assert profile.version == 1
    assert profile.patterns.tone == "neutral"
    assert profile.patterns.structure == "bulleted"


@pytest.mark.asyncio
async def test_style_profile_save_and_load(tmp_path: Path) -> None:
    profile_file = tmp_path / "style.md"
    profile = StyleProfile(str(profile_file))
    custom_patterns = LearnedPatterns(
        tone="casual",
        structure="prose",
        vocabulary={"deploy": "ship"},
        code_conventions=["uses types"],
        review_style="holistic",
    )
    await profile.save(custom_patterns)
    assert profile.version == 2

    # Reload from disk
    reloaded = StyleProfile(str(profile_file))
    assert reloaded.version == 2
    assert reloaded.patterns.tone == "casual"
    assert reloaded.patterns.structure == "prose"
    assert reloaded.patterns.vocabulary == {"deploy": "ship"}


@pytest.mark.asyncio
async def test_style_profile_update_from_samples(tmp_path: Path) -> None:
    profile_file = tmp_path / "style.md"
    profile = StyleProfile(str(profile_file))
    samples = [
        "pls ship this change thx!",
        "- item 1\n- item 2\n- item 3",
        "defect fixed in bug",
    ]
    updated = await profile.update_from_samples(samples)
    assert updated.tone == "casual"
    assert updated.vocabulary["deploy"] == "ship"
    assert profile.version == 2


@pytest.mark.asyncio
async def test_voice_matcher_heuristics() -> None:
    matcher = VoiceMatcher(llm=None)
    patterns = LearnedPatterns(
        tone="casual",
        structure="bulleted",
        vocabulary={"deploy": "ship"},
    )
    res = await matcher.evaluate(
        candidate="- ship the feature now",
        reference_samples=["ship the feature now thx"],
        patterns=patterns,
    )
    assert isinstance(res, VoiceMatchResult)
    assert res.score > 0.0
    assert res.ngram_overlap > 0.0
    assert res.vocabulary_match == 1.0
    assert res.structure_match == 1.0


def test_delegation_policy_defaults() -> None:
    policy = DelegationPolicy()
    assert policy.get_level("commit") == "draft"
    assert policy.get_level("pr_description") == "draft"
    assert policy.get_level("pr_review") == "draft_with_approval"
    assert policy.get_level("email") == "never_send_auto"


@pytest.mark.asyncio
async def test_agent_draft_commit_message(tmp_path: Path) -> None:
    profile_file = tmp_path / "style.md"
    fake_llm = FakeLLM(response="feat: add digital twin agent")
    agent = DigitalTwinAgent(
        profile_path=str(profile_file),
        llm=fake_llm,
        reference_samples=["feat: add cool feature"],
    )

    result = await agent.draft_commit_message("diff --git a/file.py b/file.py")
    assert isinstance(result, DraftResult)
    assert result.content == "feat: add digital twin agent"
    assert result.channel == "commit_messages"
    assert not result.requires_approval
    assert "drafted by twin v1" in result.attribution


@pytest.mark.asyncio
async def test_agent_draft_pr_description(tmp_path: Path) -> None:
    profile_file = tmp_path / "style.md"
    fake_llm = FakeLLM(response="## Summary\nImplementation of Digital Twin agent.")
    agent = DigitalTwinAgent(
        profile_path=str(profile_file),
        llm=fake_llm,
    )

    result = await agent.draft_pr_description(
        diff="diff --git a/b b/c", title="Digital Twin Feature"
    )
    assert isinstance(result, DraftResult)
    assert "Digital Twin" in result.content or "Summary" in result.content
    assert result.channel == "pr_descriptions"


@pytest.mark.asyncio
async def test_agent_draft_review_requires_approval(tmp_path: Path) -> None:
    profile_file = tmp_path / "style.md"
    fake_llm = FakeLLM(response="LGTM with minor nits.")
    agent = DigitalTwinAgent(
        profile_path=str(profile_file),
        llm=fake_llm,
    )

    result = await agent.draft_review(diff="diff --git a/b b/c")
    assert isinstance(result, DraftResult)
    assert result.requires_approval
    assert result.delegation_level == "draft_with_approval"


@pytest.mark.asyncio
async def test_agent_learn_and_evaluate(tmp_path: Path) -> None:
    profile_file = tmp_path / "style.md"
    agent = DigitalTwinAgent(
        profile_path=str(profile_file),
        llm=None,
    )

    samples = [
        "pls ship this change thx",
        "- feature 1 added",
    ]
    patterns = await agent.learn(samples)
    assert patterns.tone == "casual"
    assert agent.profile.version == 2

    match = await agent.evaluate_voice_match("- ship feature thx")
    assert match.score > 0.0


# --- Async contract: every public IO method must be a coroutine ---


def test_public_io_methods_are_coroutines() -> None:
    # StyleProfile file-touching public methods
    assert inspect.iscoroutinefunction(StyleProfile.load)
    assert inspect.iscoroutinefunction(StyleProfile.save)
    assert inspect.iscoroutinefunction(StyleProfile.update_from_samples)
    # DigitalTwinAgent public IO methods
    assert inspect.iscoroutinefunction(DigitalTwinAgent.learn)
    assert inspect.iscoroutinefunction(DigitalTwinAgent.draft)
    assert inspect.iscoroutinefunction(DigitalTwinAgent.send)


@pytest.mark.asyncio
async def test_style_profile_async_roundtrip(tmp_path: Path) -> None:
    profile_file = tmp_path / "style.md"
    profile = StyleProfile(str(profile_file))
    await profile.save(LearnedPatterns(tone="formal"))
    assert profile_file.exists()
    reloaded = StyleProfile(str(profile_file))
    loaded = await reloaded.load()
    assert loaded.tone == "formal"
    assert reloaded.version == 2


# --- Delegation gate enforcement ---


def test_delegation_gate_predicates() -> None:
    policy = DelegationPolicy()
    # draft channel: auto-send allowed
    assert policy.can_auto_send("commit") is True
    assert policy.requires_human_approval("commit") is False
    assert policy.is_send_forbidden("commit") is False
    # draft_with_approval channel: needs approval, not auto, not forbidden
    assert policy.can_auto_send("pr_review") is False
    assert policy.requires_human_approval("pr_review") is True
    assert policy.is_send_forbidden("pr_review") is False
    # never_send_auto channel: forbidden
    assert policy.can_auto_send("email") is False
    assert policy.requires_human_approval("email") is False
    assert policy.is_send_forbidden("email") is True


@pytest.mark.asyncio
async def test_send_never_send_auto_raises(tmp_path: Path) -> None:
    profile_file = tmp_path / "style.md"
    agent = DigitalTwinAgent(profile_path=str(profile_file), llm=None)
    result = await agent.draft("emails", "Draft an email")
    assert result.delegation_level == "never_send_auto"
    # Even with approved=True, never_send_auto cannot auto-send.
    with pytest.raises(AutoSendForbiddenError):
        await agent.send(result)
    with pytest.raises(AutoSendForbiddenError):
        await agent.send(result, approved=True)


@pytest.mark.asyncio
async def test_send_draft_with_approval_requires_token(tmp_path: Path) -> None:
    profile_file = tmp_path / "style.md"
    agent = DigitalTwinAgent(profile_path=str(profile_file), llm=None)
    result = await agent.draft("pr_reviews", "Draft a review")
    assert result.delegation_level == "draft_with_approval"
    # Without approval -> refused.
    with pytest.raises(ApprovalRequiredError):
        await agent.send(result)
    # With explicit approval -> succeeds.
    sent = await agent.send(result, approved=True)
    assert sent is result


@pytest.mark.asyncio
async def test_send_draft_level_succeeds(tmp_path: Path) -> None:
    profile_file = tmp_path / "style.md"
    agent = DigitalTwinAgent(profile_path=str(profile_file), llm=None)
    result = await agent.draft("commit_messages", "Draft a commit")
    assert result.delegation_level == "draft"
    sent = await agent.send(result)
    assert sent is result


# --- Degenerate / negative scoring ---


@pytest.mark.asyncio
async def test_empty_candidate_scores_zero() -> None:
    matcher = VoiceMatcher(llm=None)
    res = await matcher.evaluate(
        candidate="   ",
        reference_samples=["ship the feature now"],
        patterns=LearnedPatterns(),
    )
    assert res.score == 0.0
    assert res.ngram_overlap == 0.0
    assert res.vocabulary_match == 0.0
    assert res.structure_match == 0.0


@pytest.mark.asyncio
async def test_single_word_candidate_in_range() -> None:
    matcher = VoiceMatcher(llm=None)
    res = await matcher.evaluate(
        candidate="ship",
        reference_samples=["ship the feature now"],
        patterns=LearnedPatterns(),
    )
    assert 0.0 <= res.score <= 1.0


@pytest.mark.asyncio
async def test_empty_reference_samples_in_range() -> None:
    matcher = VoiceMatcher(llm=None)
    res = await matcher.evaluate(
        candidate="- ship the feature",
        reference_samples=[],
        patterns=LearnedPatterns(),
    )
    assert 0.0 <= res.score <= 1.0


@pytest.mark.asyncio
async def test_llm_judge_valid_score() -> None:
    matcher = VoiceMatcher(llm=FakeLLM(response="0.9"))
    res = await matcher.evaluate(
        candidate="- ship the feature now",
        reference_samples=["ship the feature now"],
        patterns=LearnedPatterns(),
    )
    assert 0.0 <= res.score <= 1.0
    assert res.details["llm_score"] == 0.9


@pytest.mark.asyncio
async def test_llm_judge_out_of_range_clamped() -> None:
    matcher = VoiceMatcher(llm=FakeLLM(response="1.5"))
    res = await matcher.evaluate(
        candidate="- ship the feature now",
        reference_samples=["ship the feature now"],
        patterns=LearnedPatterns(),
    )
    # 1.5 clamps to 1.0
    assert res.details["llm_score"] == 1.0
    assert 0.0 <= res.score <= 1.0


@pytest.mark.asyncio
async def test_llm_judge_non_numeric_fallback() -> None:
    matcher = VoiceMatcher(llm=FakeLLM(response="not a number"))
    res = await matcher.evaluate(
        candidate="- ship the feature now",
        reference_samples=["ship the feature now"],
        patterns=LearnedPatterns(),
    )
    # ValueError fallback -> 0.8
    assert res.details["llm_score"] == 0.8
    assert 0.0 <= res.score <= 1.0
