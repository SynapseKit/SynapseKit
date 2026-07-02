from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class ComputerActionType(str, Enum):
    """Normalized actions a computer-use model may request."""

    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    MOVE = "move"
    DRAG = "drag"
    TYPE_TEXT = "type_text"
    KEY = "key"
    HOTKEY = "hotkey"
    SCROLL = "scroll"
    WAIT = "wait"
    NAVIGATE = "navigate"
    SCREENSHOT = "screenshot"
    DONE = "done"


class SafetyDecision(str, Enum):
    """Safety gate outcome for a proposed action."""

    ALLOWED = "allowed"
    BLOCKED = "blocked"
    NEEDS_CONFIRMATION = "needs_confirmation"


@dataclass
class ComputerObservation:
    """Screen state captured before asking a provider for the next action."""

    screenshot: bytes | None = None
    text: str = ""
    app: str | None = None
    window_title: str | None = None
    url: str | None = None
    width: int | None = None
    height: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComputerAction:
    """Provider-neutral computer action."""

    type: ComputerActionType | str
    x: int | None = None
    y: int | None = None
    end_x: int | None = None
    end_y: int | None = None
    text: str | None = None
    key: str | None = None
    keys: Sequence[str] = field(default_factory=tuple)
    delta_x: int = 0
    delta_y: int = 0
    url: str | None = None
    duration: float | None = None
    reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def action_type(self) -> ComputerActionType:
        if isinstance(self.type, ComputerActionType):
            return self.type
        return ComputerActionType(str(self.type))

    @classmethod
    def done(cls, reason: str | None = None) -> ComputerAction:
        return cls(type=ComputerActionType.DONE, reason=reason)


@dataclass
class SafetyCheck:
    """Result of evaluating a single action against a policy."""

    decision: SafetyDecision
    reason: str
    action: ComputerAction
    observation: ComputerObservation | None = None

    @property
    def allowed(self) -> bool:
        return self.decision == SafetyDecision.ALLOWED


@dataclass
class ComputerStep:
    """One observe-plan-act step."""

    observation: ComputerObservation
    action: ComputerAction
    safety: SafetyCheck
    output: str | None = None
    error: str | None = None


@dataclass
class ComputerUseResult:
    """Final result from a computer-use run."""

    task: str
    completed: bool
    steps: list[ComputerStep] = field(default_factory=list)
    output: str = ""
    error: str | None = None
    recording_path: Path | None = None

    @property
    def is_error(self) -> bool:
        return self.error is not None


ConfirmationCallback = Callable[[ComputerAction, ComputerObservation], bool | Awaitable[bool]]


@runtime_checkable
class ComputerUseProvider(Protocol):
    """Provider interface used by :class:`ComputerUseAgent`."""

    async def next_action(
        self,
        task: str,
        observation: ComputerObservation,
        history: Sequence[ComputerStep],
    ) -> ComputerAction:
        """Return the next normalized action for the current screen state."""


@runtime_checkable
class ScreenProvider(Protocol):
    """Protocol for things that can observe and operate a screen."""

    async def observe(self) -> ComputerObservation:
        """Capture the current screen state."""

    async def execute(self, action: ComputerAction) -> str:
        """Apply a normalized action and return a short execution summary."""

    async def close(self) -> None:
        """Release provider resources."""
