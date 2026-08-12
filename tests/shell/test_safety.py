from __future__ import annotations

from synapsekit.shell import SafetyAnalyzer


def test_destructive_commands_are_previewed() -> None:
    assessment = SafetyAnalyzer().assess("git reset --hard HEAD")

    assert assessment.destructive
    assert "working tree" in assessment.preview


def test_similarly_named_safe_command_is_not_rm() -> None:
    assessment = SafetyAnalyzer().assess("rmdir --help")

    assert assessment.destructive


def test_read_only_command_is_not_destructive() -> None:
    assessment = SafetyAnalyzer().assess("git status --short")

    assert not assessment.destructive
