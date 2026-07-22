from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest

from synapsekit.llm.base import BaseLLM, LLMConfig
from synapsekit.twin import (
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


def test_style_profile_save_and_load(tmp_path: Path) -> None:
    profile_file = tmp_path / "style.md"
    profile = StyleProfile(str(profile_file))
    custom_patterns = LearnedPatterns(
        tone="casual",
        structure="prose",
        vocabulary={"deploy": "ship"},
        code_conventions=["uses types"],
        review_style="holistic",
    )
    profile.save(custom_patterns)
    assert profile.version == 2

    # Reload from disk
    reloaded = StyleProfile(str(profile_file))
    assert reloaded.version == 2
    assert reloaded.patterns.tone == "casual"
    assert reloaded.patterns.structure == "prose"
    assert reloaded.patterns.vocabulary == {"deploy": "ship"}


def test_style_profile_update_from_samples(tmp_path: Path) -> None:
    profile_file = tmp_path / "style.md"
    profile = StyleProfile(str(profile_file))
    samples = [
        "pls ship this change thx!",
        "- item 1\n- item 2\n- item 3",
        "defect fixed in bug",
    ]
    updated = profile.update_from_samples(samples)
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
