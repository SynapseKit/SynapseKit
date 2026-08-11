"""Fail-closed safety policy for agent-generated shell commands."""

from __future__ import annotations

import shlex
from dataclasses import dataclass

from .types import SafetyAssessment


class SafetyError(RuntimeError):
    """Raised when a command cannot pass the shell safety policy."""


@dataclass(frozen=True)
class SafetyPolicy:
    """Policy knobs intentionally biased toward user confirmation."""

    allow_destructive: bool = True
    require_signed_approval: bool = True
    max_command_length: int = 8_192


class SafetyAnalyzer:
    """Classify commands before they reach the subprocess executor."""

    _DESTRUCTIVE = (
        ("git reset", "rewrites the working tree or index"),
        ("git clean", "deletes untracked files"),
        ("git push --force", "rewrites a remote branch"),
        ("git push -f", "rewrites a remote branch"),
        ("git branch -D", "force-deletes a branch"),
        ("git rm", "removes tracked files"),
        ("git restore", "overwrites working tree changes"),
        ("git checkout --", "overwrites working tree changes"),
        ("git push --delete", "deletes a remote branch"),
        ("rm", "removes files or directories"),
        ("remove-item", "removes files or directories"),
        ("rmdir", "removes directories"),
        ("del", "removes files"),
        ("docker system prune", "removes Docker resources"),
        ("docker volume rm", "removes Docker volumes"),
        ("kubectl delete", "deletes cluster resources"),
        ("terraform destroy", "destroys infrastructure"),
    )

    def __init__(self, policy: SafetyPolicy | None = None) -> None:
        self.policy = policy or SafetyPolicy()

    def assess(self, command: str) -> SafetyAssessment:
        if len(command) > self.policy.max_command_length:
            raise SafetyError(
                f"command is too long ({len(command)} characters; "
                f"limit is {self.policy.max_command_length})"
            )
        normalized = " ".join(command.casefold().split())
        reasons: list[str] = []
        try:
            argv = shlex.split(command, posix=True)
        except ValueError as exc:
            raise SafetyError(f"invalid command: {exc}") from exc
        if not argv:
            raise SafetyError("empty command")

        executable = argv[0].casefold().rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
        for marker, reason in self._DESTRUCTIVE:
            if marker in normalized and (marker != "rm" or executable in {"rm", "rm.exe"}):
                reasons.append(reason)
        if executable in {"mv", "move", "move-item", "copy", "cp", "copy-item"} and any(
            flag in normalized for flag in (" -f", " -force", " --force")
        ):
            reasons.append("forcefully overwrites or moves existing data")
        if any(token in normalized for token in (">", ">>", "2>", " 2>>")):
            reasons.append("overwrites or redirects a file")
        if "--force" in argv or "-f" in argv[1:]:
            reasons.append("uses a force flag")
        destructive = bool(reasons)
        preview = self._preview(argv, reasons)
        return SafetyAssessment(
            destructive=destructive, reasons=tuple(dict.fromkeys(reasons)), preview=preview
        )

    @staticmethod
    def _preview(argv: list[str], reasons: list[str]) -> str:
        rendered = " ".join(shlex.quote(part) for part in argv)
        if not reasons:
            return rendered
        return f"{rendered}\nReasons: " + "; ".join(dict.fromkeys(reasons))
