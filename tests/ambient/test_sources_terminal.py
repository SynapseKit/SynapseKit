"""Tests for TerminalSourcePlugin history-file tailing."""

from __future__ import annotations

from pathlib import Path

import pytest

from synapsekit.ambient.sources.terminal import TerminalSourcePlugin


@pytest.mark.asyncio
async def test_first_poll_establishes_baseline_without_emitting(tmp_path: Path) -> None:
    history = tmp_path / "history.txt"
    history.write_text("git status\nls -la\n", encoding="utf-8")

    plugin = TerminalSourcePlugin(history_path=history)
    events = await plugin.poll()
    assert events == []


@pytest.mark.asyncio
async def test_second_poll_only_returns_new_lines(tmp_path: Path) -> None:
    history = tmp_path / "history.txt"
    history.write_text("git status\n", encoding="utf-8")

    plugin = TerminalSourcePlugin(history_path=history)
    await plugin.poll()

    with open(history, "a", encoding="utf-8") as f:
        f.write("rm -rf build\n")

    events = await plugin.poll()
    assert [e.text for e in events] == ["rm -rf build"]
    assert events[0].source == "terminal"
    assert events[0].kind == "command"


@pytest.mark.asyncio
async def test_no_new_lines_returns_empty(tmp_path: Path) -> None:
    history = tmp_path / "history.txt"
    history.write_text("git status\n", encoding="utf-8")

    plugin = TerminalSourcePlugin(history_path=history)
    await plugin.poll()
    assert await plugin.poll() == []


@pytest.mark.asyncio
async def test_missing_history_file_returns_empty(tmp_path: Path) -> None:
    plugin = TerminalSourcePlugin(history_path=tmp_path / "nonexistent.txt")
    assert await plugin.poll() == []
    assert await plugin.poll() == []
