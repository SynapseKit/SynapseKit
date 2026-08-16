"""Tests for GitSourcePlugin against a real git repo."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from synapsekit.ambient.sources.git import GitSourcePlugin


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-b", "main", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git("add", ".", cwd=repo)
    _git("commit", "-m", "initial", cwd=repo)
    return repo


@pytest.mark.asyncio
async def test_poll_reports_clean_repo_as_not_dirty(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    plugin = GitSourcePlugin(repo)
    events = await plugin.poll()
    assert len(events) == 1
    assert events[0].metadata["dirty"] is False
    assert events[0].metadata["branch"] == "main"


@pytest.mark.asyncio
async def test_poll_reports_dirty_repo(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    plugin = GitSourcePlugin(repo)
    await plugin.poll()  # baseline clean-state event

    (repo / "README.md").write_text("changed\n", encoding="utf-8")
    events = await plugin.poll()

    assert len(events) == 1
    assert events[0].metadata["dirty"] is True
    assert "README.md" in events[0].metadata["dirty_files"][0]


@pytest.mark.asyncio
async def test_poll_dedupes_unchanged_status(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    plugin = GitSourcePlugin(repo)
    await plugin.poll()

    events = await plugin.poll()
    assert events == []


@pytest.mark.asyncio
async def test_poll_on_non_repo_returns_empty_and_disables(tmp_path: Path) -> None:
    not_a_repo = tmp_path / "not_a_repo"
    not_a_repo.mkdir()
    plugin = GitSourcePlugin(not_a_repo)

    assert await plugin.poll() == []
    assert plugin._unavailable is True
    assert await plugin.poll() == []
