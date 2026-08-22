"""Unit tests for Code Archaeology Agent (Issue #744)."""

from __future__ import annotations

import subprocess
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

import synapsekit
from synapsekit.archaeology import (
    ArchaeologyAgent,
    ArchaeologyResult,
    CausalClaim,
    CausalLinker,
    Citation,
    EvolutionDiff,
    EvolutionSnapshot,
    SourceConfig,
    TimelineEvent,
    TimelineReconstructor,
)
from synapsekit.llm.base import BaseLLM, LLMConfig
from synapsekit.loaders.base import Document


class FakeLLM(BaseLLM):
    def __init__(self, response: str = "") -> None:
        super().__init__(LLMConfig(model="fake", api_key="", provider="fake"))
        self.response = response

    async def stream(self, prompt: str, **kw) -> AsyncGenerator[str]:
        yield self.response


class FakeLoader:
    """Hand-written stand-in for a loader's `.aload()` boundary."""

    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs

    async def aload(self) -> list[Document]:
        return self._docs


@dataclass
class FakeVerificationResult:
    verified: bool


class FakeVerifier:
    """Hand-written stand-in for a NeuroSymbolicAgent-shaped verifier."""

    def __init__(self, verified: bool) -> None:
        self._verified = verified

    async def solve(self, problem: str) -> FakeVerificationResult:
        return FakeVerificationResult(verified=self._verified)


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


async def test_timeline_slack_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    tr = TimelineReconstructor(repo_path=tmp_path)
    slack_doc = Document(
        text="Discussing why AgentRegistry was introduced in architecture review",
        metadata={"timestamp": "1700000000.0", "channel": "C123", "user": "U456"},
    )
    monkeypatch.setattr(
        "synapsekit.loaders.slack.SlackLoader.aload",
        FakeLoader([slack_doc]).aload,
    )
    events = await tr.reconstruct(
        "AgentRegistry",
        include_git=False,
        slack_bot_token="xoxb-fake",
        slack_channel_ids=["C123"],
    )
    assert len(events) == 1
    assert events[0].source_type == "slack"
    assert "architecture review" in events[0].summary


async def test_timeline_email_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    tr = TimelineReconstructor(repo_path=tmp_path)
    email_doc = Document(
        text="AgentRegistry proposal thread body text",
        metadata={
            "subject": "Proposal: AgentRegistry",
            "from": "alice@synapsekit.dev",
            "date": "Mon, 15 Jan 2024 10:00:00 +0000",
            "email_id": "101",
        },
    )
    monkeypatch.setattr(
        "synapsekit.loaders.email.EmailLoader.aload",
        FakeLoader([email_doc]).aload,
    )
    events = await tr.reconstruct(
        "AgentRegistry",
        include_git=False,
        email_imap_server="imap.fake.com",
        email_address="me@fake.com",
        email_password="fake",
    )
    assert len(events) == 1
    assert events[0].source_type == "email"
    assert events[0].timestamp == datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
    assert "Proposal: AgentRegistry" in events[0].summary


async def test_causal_linker_parses_claims():
    llm_response = (
        "CAUSE: PR #718 introduced AgentRegistry\n"
        "EFFECT: Agents could be dynamically dispatched\n"
        "CONFIDENCE: 0.9\n"
        "REASONING: Commit message explicitly states purpose\n"
        "---\n"
        "CAUSE: Feature request for multi-agent systems\n"
        "EFFECT: AgentRegistry added register method\n"
        "CONFIDENCE: 0.7\n"
        "REASONING: Follow-up PR extended the class\n"
        "---"
    )
    llm = FakeLLM(response=llm_response)
    linker = CausalLinker(llm, min_citations=0)

    c = Citation(source_type="git", reference="commit abc", content_preview="AgentRegistry")
    events = [
        TimelineEvent(
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            source_type="git",
            summary="Created AgentRegistry (#718)",
            citations=[c],
        ),
    ]
    claims = await linker.link(events, "AgentRegistry")
    assert len(claims) == 2
    assert claims[0].cause == "PR #718 introduced AgentRegistry"
    assert claims[0].confidence == 0.9


async def test_causal_linker_min_citations_filter():
    llm_response = (
        "CAUSE: Unknown cause\n"
        "EFFECT: Unknown effect\n"
        "CONFIDENCE: 0.3\n"
        "REASONING: Weak evidence\n"
        "---"
    )
    llm = FakeLLM(response=llm_response)
    linker = CausalLinker(llm, min_citations=2)

    events = [
        TimelineEvent(
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            source_type="git",
            summary="Some event",
            citations=[],
        ),
    ]
    claims = await linker.link(events, "test")
    assert len(claims) == 0  # Filtered out due to insufficient citations


async def test_causal_linker_empty_events():
    llm = FakeLLM()
    linker = CausalLinker(llm)
    claims = await linker.link([], "test")
    assert claims == []


async def test_causal_linker_with_verifier():
    llm_response = (
        "CAUSE: Issue #718 required dynamic agents\n"
        "EFFECT: AgentRegistry was created\n"
        "CONFIDENCE: 0.95\n"
        "REASONING: Strong evidence from issue discussion\n"
        "---"
    )
    llm = FakeLLM(response=llm_response)

    linker = CausalLinker(llm, verifier=FakeVerifier(verified=True), min_citations=0)
    c = Citation(
        source_type="git",
        reference="commit abc",
        content_preview="Issue #718 required dynamic agents",
    )
    events = [
        TimelineEvent(
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            source_type="git",
            summary="AgentRegistry created for #718",
            citations=[c],
        ),
    ]
    claims = await linker.link(events, "AgentRegistry")
    assert len(claims) == 1
    assert claims[0].verified is True


async def test_evolution_diff_trace(git_repo: Path):
    ed = EvolutionDiff(repo_path=git_repo)
    snapshots = await ed.trace("agent.py")
    assert len(snapshots) >= 1
    assert all(isinstance(s, EvolutionSnapshot) for s in snapshots)
    assert snapshots[0].version_hash


async def test_evolution_diff_trace_empty(git_repo: Path):
    ed = EvolutionDiff(repo_path=git_repo)
    snapshots = await ed.trace("nonexistent_file.py")
    assert snapshots == []


async def test_evolution_diff_trace_unmatched_nl_query_returns_empty(git_repo: Path):
    """Regression: an unmatched natural-language query must not fall back to
    returning the entire unfiltered repo history (the path ArchaeologyAgent.explain()
    exercises for every query)."""
    ed = EvolutionDiff(repo_path=git_repo)
    snapshots = await ed.trace("why does something totally unrelated exist")
    assert snapshots == []


async def test_archaeology_agent_explain_git_only(git_repo: Path):
    """Full integration test with git-only sources."""
    llm = FakeLLM(response="AgentRegistry was created for dynamic agent dispatch.")
    sources = SourceConfig(repo_path=str(git_repo), include_git=True, min_citations_per_claim=0)
    agent = ArchaeologyAgent(sources=sources, llm=llm)
    result = await agent.explain("Why does AgentRegistry exist?")

    assert isinstance(result, ArchaeologyResult)
    assert result.query == "Why does AgentRegistry exist?"
    assert len(result.timeline) >= 1
    assert len(result.evolution) >= 1
    assert isinstance(result.narrative, str)
    assert len(result.narrative) > 0


async def test_archaeology_agent_no_llm(git_repo: Path):
    """Agent works without LLM (causal linking and narrative disabled)."""
    sources = SourceConfig(repo_path=str(git_repo))
    agent = ArchaeologyAgent(sources=sources, llm=None)
    # Force llm to None to test graceful degradation
    agent.llm = None
    agent.causal_linker = None
    result = await agent.explain("AgentRegistry")

    assert isinstance(result, ArchaeologyResult)
    assert len(result.timeline) >= 1
    assert result.narrative == ""
    assert result.causes == []


async def test_archaeology_agent_result_markdown(git_repo: Path):
    """Verify the result can be rendered as markdown."""
    llm = FakeLLM(response="Test narrative.")
    sources = SourceConfig(repo_path=str(git_repo), min_citations_per_claim=0)
    agent = ArchaeologyAgent(sources=sources, llm=llm)
    result = await agent.explain("AgentRegistry")
    md = result.to_markdown()
    assert "# Code Archaeology:" in md


def test_archaeology_exports():
    """Verify archaeology classes are accessible from top-level synapsekit."""
    assert hasattr(synapsekit, "ArchaeologyAgent")
    assert hasattr(synapsekit, "ArchaeologyResult")
    assert hasattr(synapsekit, "CausalClaim")
    assert hasattr(synapsekit, "Citation")
    assert hasattr(synapsekit, "TimelineEvent")
    assert hasattr(synapsekit, "SourceConfig")
    assert hasattr(synapsekit, "TimelineReconstructor")
    assert hasattr(synapsekit, "EvolutionDiff")
    assert hasattr(synapsekit, "EvolutionSnapshot")
    assert hasattr(synapsekit.archaeology, "CausalLinker")
