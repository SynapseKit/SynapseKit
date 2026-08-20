"""Unit tests for Code Archaeology Agent (Issue #744)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from synapsekit.archaeology.types import (
    ArchaeologyResult,
    CausalClaim,
    Citation,
    EvolutionSnapshot,
    SourceConfig,
    TimelineEvent,
)


def test_citation_construction():
    c = Citation(
        source_type="git",
        reference="commit abc1234",
        content_preview="Initial commit",
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
    )
    assert c.source_type == "git"
    assert c.reference == "commit abc1234"
    assert c.timestamp is not None


def test_timeline_event_construction():
    c = Citation(source_type="git", reference="commit abc", content_preview="test")
    event = TimelineEvent(
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        source_type="git",
        summary="Added file.py",
        citations=[c],
    )
    assert event.source_type == "git"
    assert len(event.citations) == 1


def test_causal_claim_construction():
    claim = CausalClaim(
        cause="PR #42 added caching",
        effect="Response time improved 3x",
        confidence=0.85,
        verified=True,
        reasoning="Commit message explicitly mentions perf improvement",
    )
    assert claim.verified is True
    assert claim.confidence == 0.85


def test_evolution_snapshot_construction():
    snap = EvolutionSnapshot(
        version_hash="abc1234",
        date=datetime(2024, 6, 15, tzinfo=UTC),
        diff_summary="modified agent.py [AgentRegistry] (+20/-5)",
        reason="Add register method to AgentRegistry (#719)",
    )
    assert snap.version_hash == "abc1234"


def test_archaeology_result_to_markdown():
    c = Citation(source_type="git", reference="commit abc", content_preview="test commit")
    result = ArchaeologyResult(
        query="Why does AgentRegistry exist?",
        timeline=[
            TimelineEvent(
                timestamp=datetime(2024, 1, 1, tzinfo=UTC),
                source_type="git",
                summary="Created AgentRegistry",
                citations=[c],
            ),
        ],
        causes=[
            CausalClaim(
                cause="Need for dynamic agent dispatch",
                effect="AgentRegistry was created",
                confidence=0.9,
                citations=[c],
                verified=True,
            ),
        ],
        evolution=[
            EvolutionSnapshot(
                version_hash="abc1234",
                date=datetime(2024, 1, 1, tzinfo=UTC),
                diff_summary="added agent_registry.py",
                reason="Initial creation",
                citations=[c],
            ),
        ],
        narrative="AgentRegistry was created to enable dynamic agent dispatch.",
    )
    md = result.to_markdown()
    assert "AgentRegistry" in md
    assert "## Timeline" in md
    assert "## Causal Chain" in md
    assert "## Evolution History" in md
    assert "## Citations" in md


def test_archaeology_result_all_citations_deduplication():
    c1 = Citation(source_type="git", reference="commit abc", content_preview="test")
    c2 = Citation(source_type="slack", reference="slack://C1/123", content_preview="msg")
    result = ArchaeologyResult(
        query="test",
        timeline=[
            TimelineEvent(
                timestamp=datetime(2024, 1, 1, tzinfo=UTC),
                source_type="git",
                summary="test",
                citations=[c1, c2],
            ),
        ],
        causes=[
            CausalClaim(cause="a", effect="b", confidence=0.5, citations=[c1]),
        ],
    )
    all_cites = result.all_citations
    assert len(all_cites) == 2  # c1 deduplicated


def test_source_config_defaults():
    cfg = SourceConfig()
    assert cfg.repo_path == "."
    assert cfg.include_git is True
    assert cfg.min_citations_per_claim == 2
    assert cfg.slack_bot_token is None
