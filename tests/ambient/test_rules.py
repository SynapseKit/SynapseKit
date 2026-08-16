"""Tests for the ambient rule-based trigger policy."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from synapsekit.ambient.events import AmbientEvent, AmbientState
from synapsekit.ambient.rules import evaluate


def _event(text: str, source: str = "terminal") -> AmbientEvent:
    return AmbientEvent(source=source, kind="command", text=text, timestamp=datetime.now(UTC))


def _dirty_state() -> AmbientState:
    return AmbientState(git_dirty=True, dirty_files=("foo.py",), branch="main")


def _clean_state() -> AmbientState:
    return AmbientState(git_dirty=False)


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf build",
        "rm -fr build",
        "Remove-Item -Recurse -Force somefile",
        "Remove-Item -Force -Recurse somefile",
        "git reset --hard",
        "git push --force origin main",
        "git push -f origin main",
        "git clean -fd",
    ],
)
def test_risky_command_fires_when_repo_dirty(command: str) -> None:
    intervention = evaluate(_event(command), _dirty_state())
    assert intervention is not None
    assert intervention.confidence > 0
    assert command.strip() in intervention.message


@pytest.mark.parametrize(
    "command",
    ["rm -rf build", "git reset --hard", "git push --force"],
)
def test_risky_command_does_not_fire_when_repo_clean(command: str) -> None:
    assert evaluate(_event(command), _clean_state()) is None


def test_unrelated_command_does_not_fire() -> None:
    assert evaluate(_event("git status"), _dirty_state()) is None
    assert evaluate(_event("ls -la"), _dirty_state()) is None


def test_non_terminal_source_never_fires() -> None:
    event = _event("rm -rf build", source="git")
    assert evaluate(event, _dirty_state()) is None
