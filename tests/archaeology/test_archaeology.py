"""Unit tests for Code Archaeology Agent (Issue #744)."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from synapsekit.archaeology.timeline_reconstructor import TimelineReconstructor
from synapsekit.archaeology.types import (
    ArchaeologyResult,
    CausalClaim,
    Citation,
    EvolutionSnapshot,
    SourceConfig,
    TimelineEvent,
)
from synapsekit.loaders.base import Document


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


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository with multi-commit history."""
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    (tmp_path / "agent.py").write_text(
        "class AgentRegistry:\n    def __init__(self):\n        pass\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial AgentRegistry (#718)"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    (tmp_path / "agent.py").write_text(
        "class AgentRegistry:\n    def __init__(self):\n        pass\n\n    def register(self):\n        pass\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Add register method (#719)"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    return tmp_path


async def test_timeline_git_events(git_repo: Path):
    tr = TimelineReconstructor(repo_path=git_repo)
    events = await tr.reconstruct("AgentRegistry", include_git=True)
    assert len(events) >= 1
    assert all(e.source_type == "git" for e in events)
    assert any("AgentRegistry" in e.summary for e in events)


async def test_timeline_events_sorted_chronologically(git_repo: Path):
    tr = TimelineReconstructor(repo_path=git_repo)
    events = await tr.reconstruct("AgentRegistry", include_git=True)
    timestamps = [e.timestamp for e in events]
    assert timestamps == sorted(timestamps)


async def test_timeline_markdown_events(tmp_path: Path):
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    (notes_dir / "design.md").write_text(
        "AgentRegistry was designed for dynamic dispatch",
        encoding="utf-8",
    )
    tr = TimelineReconstructor(repo_path=tmp_path)
    events = await tr.reconstruct(
        "AgentRegistry",
        include_git=False,
        markdown_roots=[notes_dir],
    )
    assert len(events) >= 1
    assert events[0].source_type == "markdown"


async def test_timeline_empty_query(git_repo: Path):
    tr = TimelineReconstructor(repo_path=git_repo)
    events = await tr.reconstruct("xyznonexistent", include_git=True)
    assert isinstance(events, list)


async def test_timeline_slack_events(tmp_path: Path):
    tr = TimelineReconstructor(repo_path=tmp_path)
    mock_slack_doc = Document(
        text="Discussing why AgentRegistry was introduced in architecture review",
        metadata={"timestamp": "1700000000.0", "channel": "C123", "user": "U456"},
    )
    with patch(
        "synapsekit.loaders.slack.SlackLoader.aload",
        new=AsyncMock(return_value=[mock_slack_doc]),
    ):
        events = await tr.reconstruct(
            "AgentRegistry",
            include_git=False,
            slack_bot_token="xoxb-fake",
            slack_channel_ids=["C123"],
        )
        assert len(events) == 1
        assert events[0].source_type == "slack"
        assert "architecture review" in events[0].summary


async def test_timeline_email_events(tmp_path: Path):
    tr = TimelineReconstructor(repo_path=tmp_path)
    mock_email_doc = Document(
        text="AgentRegistry proposal thread body text",
        metadata={
            "subject": "Proposal: AgentRegistry",
            "from": "alice@synapsekit.dev",
            "date": "2024-01-15T10:00:00+00:00",
            "email_id": "101",
        },
    )
    with patch(
        "synapsekit.loaders.email.EmailLoader.aload",
        new=AsyncMock(return_value=[mock_email_doc]),
    ):
        events = await tr.reconstruct(
            "AgentRegistry",
            include_git=False,
            email_imap_server="imap.fake.com",
            email_address="me@fake.com",
            email_password="fake",
        )
        assert len(events) == 1
        assert events[0].source_type == "email"
        assert "Proposal: AgentRegistry" in events[0].summary
