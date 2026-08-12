"""Public contracts for SynapseKit's agent-aware shell."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class ShellKind(str, Enum):
    """Shell dialects supported by the integration plugins."""

    BASH = "bash"
    ZSH = "zsh"
    FISH = "fish"
    POWERSHELL = "powershell"

    @classmethod
    def parse(cls, value: str) -> ShellKind:
        normalized = value.casefold().strip()
        if normalized in {"pwsh", "ps", "powershell.exe"}:
            normalized = cls.POWERSHELL.value
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError(f"unsupported shell: {value!r}") from exc


class SegmentKind(str, Enum):
    """Whether an input segment is executable shell text or natural language."""

    SHELL = "shell"
    NATURAL_LANGUAGE = "natural_language"


@dataclass(frozen=True)
class InputSegment:
    """A source-preserving part of a mixed shell line."""

    text: str
    kind: SegmentKind
    start: int
    end: int


@dataclass(frozen=True)
class ParsedInput:
    """Result of lexing one line."""

    raw: str
    segments: tuple[InputSegment, ...]

    @property
    def has_natural_language(self) -> bool:
        return any(segment.kind is SegmentKind.NATURAL_LANGUAGE for segment in self.segments)


@dataclass(frozen=True)
class ShellCommand:
    """A shell command represented as argv, never as an executable shell string."""

    argv: tuple[str, ...]
    raw: str
    connector: Literal["", "&&", "||", ";", "|"] = ""

    @property
    def executable(self) -> str:
        return self.argv[0] if self.argv else ""


@dataclass(frozen=True)
class ShellContext:
    """Minimal, redaction-friendly context supplied to planners."""

    cwd: str
    platform: str
    shell: ShellKind
    git_branch: str | None = None
    git_status: str | None = None
    mesh_hits: tuple[dict[str, Any], ...] = ()
    ambient: dict[str, Any] = field(default_factory=dict)
    history: tuple[dict[str, Any], ...] = ()
    environment: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cwd": self.cwd,
            "platform": self.platform,
            "shell": self.shell.value,
            "git_branch": self.git_branch,
            "git_status": self.git_status,
            "mesh_hits": list(self.mesh_hits),
            "ambient": dict(self.ambient),
            "history": list(self.history),
            "environment": dict(self.environment),
        }


@dataclass(frozen=True)
class PlannedStep:
    """A planner output that still must pass the safety analyzer."""

    command: str
    explanation: str = ""
    source: Literal["shell", "rule", "llm"] = "shell"
    confidence: float = 1.0
    connector: Literal["", "&&", "||", ";", "|"] = ""


@dataclass(frozen=True)
class ShellPlan:
    """Validated plan shown to a user before execution."""

    input_text: str
    steps: tuple[PlannedStep, ...]
    summary: str
    warnings: tuple[str, ...] = ()

    @property
    def requires_confirmation(self) -> bool:
        return any(step_is_destructive(step.command) for step in self.steps)


@dataclass(frozen=True)
class SafetyAssessment:
    """Classification of a command and its reasons."""

    destructive: bool
    reasons: tuple[str, ...] = ()
    preview: str = ""


@dataclass(frozen=True)
class CommandResult:
    """Captured result of one direct subprocess invocation."""

    command: str
    stdout: str
    stderr: str
    exit_code: int | None
    duration_seconds: float
    timed_out: bool = False
    skipped: bool = False

    @property
    def ok(self) -> bool:
        return not self.timed_out and not self.skipped and self.exit_code == 0


@dataclass(frozen=True)
class ShellRunResult:
    """Complete execution result returned by :class:`ShellSession`."""

    plan: ShellPlan
    commands: tuple[CommandResult, ...]
    aborted: bool = False
    error: str | None = None
    audit_path: str | None = None

    @property
    def ok(self) -> bool:
        return (
            not self.aborted and self.error is None and all(result.ok for result in self.commands)
        )


def step_is_destructive(command: str) -> bool:
    """Small dependency-free helper used by the immutable plan contract."""

    lowered = command.casefold().strip()
    return any(
        marker in lowered
        for marker in (
            "git reset",
            "git clean",
            "git push --force",
            "git push -f",
            "rm ",
            "remove-item",
            " rmdir",
            " del ",
            "docker system prune",
            "docker volume rm",
            "kubectl delete",
        )
    )
