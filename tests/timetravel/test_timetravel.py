"""Unit tests for Time-Travel Codebase subpackage (Issue #746)."""

from __future__ import annotations

import subprocess
from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import pytest

from synapsekit.llm.base import BaseLLM, LLMConfig
from synapsekit.timetravel import (
    AsOf,
    CommitInfo,
    DiffNarrativeGenerator,
    DriftCandidate,
    DriftDetector,
    EvolutionEntry,
    EvolutionIndex,
    GitBackend,
    TimeTravelAgent,
)


class FakeLLM(BaseLLM):
    def __init__(self, response: str = "Test narrative response") -> None:
        super().__init__(
            LLMConfig(
                model="fake",
                api_key="",
                provider="fake",
            )
        )
        self.response = response
        self.prompts: list[str] = []

    async def stream(self, prompt: str, **kw) -> AsyncGenerator[str]:
        self.prompts.append(prompt)
        yield self.response


@pytest.fixture
def git_repo(tmp_path):
    """Create a temporary git repository with multi-commit history."""
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True)

    # Commit 1: Initial file
    (tmp_path / "agent.py").write_text(
        "class AgentRegistry:\n    def __init__(self):\n        pass\n"
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial AgentRegistry (#718)"], cwd=tmp_path, check=True
    )

    # Commit 2: Add methods
    (tmp_path / "agent.py").write_text(
        "class AgentRegistry:\n"
        "    def __init__(self):\n"
        "        pass\n\n"
        "    def register(self):\n"
        "        pass\n"
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Add register method to AgentRegistry (#719)"],
        cwd=tmp_path,
        check=True,
    )

    # Commit 3: Add helper file
    (tmp_path / "helper.py").write_text("def helper_func():\n    return 42\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "Add helper module"], cwd=tmp_path, check=True)

    return tmp_path


def test_git_backend_log(git_repo):
    backend = GitBackend(git_repo)
    commits = backend.log()

    assert len(commits) == 3
    assert commits[0].subject == "Add helper module"
    assert "Initial AgentRegistry (#718)" in commits[-1].subject
    assert commits[0].author == "Test User"


def test_git_backend_diff_and_show(git_repo):
    backend = GitBackend(git_repo)
    commits = backend.log()
    c_latest = commits[0].hash
    c_earliest = commits[-1].hash

    diff_text = backend.diff(c_earliest, c_latest)
    assert diff_text != ""

    content = backend.show(c_earliest, "agent.py")
    assert "class AgentRegistry:" in content
    assert "def register" not in content


def test_git_backend_find_commit_at(git_repo):
    backend = GitBackend(git_repo)
    now = datetime.now(UTC)

    commit_hash = backend.find_commit_at(now)
    assert commit_hash != ""


def test_git_backend_list_files(git_repo):
    backend = GitBackend(git_repo)
    files = backend.list_files()

    assert "agent.py" in files
    assert "helper.py" in files


def test_evolution_index_build_and_query(git_repo):
    backend = GitBackend(git_repo)
    index = EvolutionIndex(backend)

    entries = index.build()
    assert len(entries) > 0

    agent_entries = index.query("AgentRegistry")
    assert len(agent_entries) >= 2
    assert agent_entries[0].symbol == "AgentRegistry"
    assert agent_entries[0].pr_number in (718, 719)


def test_evolution_index_timeline(git_repo):
    backend = GitBackend(git_repo)
    index = EvolutionIndex(backend)

    timeline = index.timeline("agent.py")
    assert len(timeline) >= 2
    assert timeline[0].commit.date <= timeline[-1].commit.date


def test_drift_detector_detect(git_repo):
    backend = GitBackend(git_repo)
    detector = DriftDetector(backend)

    candidates = detector.detect(min_age_days=0)
    assert isinstance(candidates, list)
    if candidates:
        assert isinstance(candidates[0], DriftCandidate)
        assert candidates[0].symbol != ""


@pytest.mark.asyncio
async def test_narrative_generator_heuristic():
    generator = DiffNarrativeGenerator()

    dummy_commit = CommitInfo(
        hash="1234567890abcdef",
        author="Alice",
        date=datetime.now(UTC),
        subject="Add feature (#101)",
        body="Detailed body",
        files_changed=["main.py"],
    )
    entry = EvolutionEntry(
        file_path="main.py",
        symbol="MainClass",
        commit=dummy_commit,
        diff_snippet="+ class MainClass:\n+     pass",
        change_type="added",
        lines_added=2,
        lines_removed=0,
        pr_number=101,
    )

    narrative = await generator.generate([entry], "how has MainClass changed?")
    assert "Code Evolution Summary" in narrative
    assert "Alice" in narrative
    assert "MainClass" in narrative
    assert "#101" in narrative


@pytest.mark.asyncio
async def test_narrative_generator_llm():
    fake_llm = FakeLLM("The class evolved from simple to complex.")
    generator = DiffNarrativeGenerator(llm=fake_llm)

    dummy_commit = CommitInfo(
        hash="1234567890abcdef",
        author="Bob",
        date=datetime.now(UTC),
        subject="Refactor core",
        body="",
        files_changed=["core.py"],
    )
    entry = EvolutionEntry(
        file_path="core.py",
        symbol="CoreEngine",
        commit=dummy_commit,
        diff_snippet="+ def run(): pass",
        change_type="modified",
        lines_added=1,
        lines_removed=0,
    )

    narrative = await generator.generate([entry], "CoreEngine evolution", llm=fake_llm)
    assert narrative == "The class evolved from simple to complex."
    assert len(fake_llm.prompts) == 1


@pytest.mark.asyncio
async def test_time_travel_agent_query(git_repo):
    agent = TimeTravelAgent(repo=git_repo)
    answer = await agent.query("how has AgentRegistry changed?")

    assert "AgentRegistry" in answer or "Code Evolution" in answer


@pytest.mark.asyncio
async def test_time_travel_agent_as_of(git_repo):
    agent = TimeTravelAgent(repo=git_repo)
    now = datetime.now(UTC)

    as_of = agent.as_of(now)
    assert isinstance(as_of, AsOf)
    assert as_of.commit != ""

    res = await as_of.query("how did auth work?")
    assert "Codebase State As Of" in res


@pytest.mark.asyncio
async def test_time_travel_agent_drift_and_timeline(git_repo):
    agent = TimeTravelAgent(repo=git_repo)

    drift = await agent.detect_drift("AgentRegistry")
    assert isinstance(drift, list)

    timeline = await agent.timeline("AgentRegistry")
    assert len(timeline) >= 2
