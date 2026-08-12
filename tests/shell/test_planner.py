from __future__ import annotations

import asyncio
from pathlib import Path

from synapsekit.shell import LLMShellPlanner, RuleBasedPlanner, ShellContext, ShellKind


def _context() -> ShellContext:
    return ShellContext(cwd=str(Path.cwd()), platform="test", shell=ShellKind.BASH)


def test_rule_planner_translates_common_status_request() -> None:
    steps = asyncio.run(RuleBasedPlanner().plan("show me git status", _context()))

    assert steps[0].command == "git status --short --branch"
    assert steps[0].source == "rule"


def test_llm_planner_requires_strict_json() -> None:
    class FakeLLM:
        async def generate(self, _prompt: str, **_kwargs: object) -> str:
            return '{"steps":[{"command":"echo hello","explanation":"say hello"}]}'

    steps = asyncio.run(LLMShellPlanner(FakeLLM()).plan("say hello", _context()))

    assert steps[0].command == "echo hello"
    assert steps[0].source == "llm"
