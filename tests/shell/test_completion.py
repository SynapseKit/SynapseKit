from __future__ import annotations

import asyncio
from pathlib import Path

from synapsekit.shell import CompletionEngine, SafetyAnalyzer, all_adapters


def test_each_shell_plugin_calls_completion_command() -> None:
    scripts = [adapter.init_script() for adapter in all_adapters()]

    assert all("synapsekit shell complete" in script for script in scripts)


def test_local_completion_returns_executable_prefix() -> None:
    values = asyncio.run(CompletionEngine().complete("python", cwd=Path.cwd()))

    assert "python" in values


def test_git_rm_is_destructive() -> None:
    assessment = SafetyAnalyzer().assess("git rm important.txt")

    assert assessment.destructive
    assert "tracked files" in assessment.preview
