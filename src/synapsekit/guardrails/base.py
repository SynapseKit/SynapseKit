"""The :class:`Guard` base class and the context passed to every check."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .types import GuardFinding, Mode, Stage


@dataclass
class GuardContext:
    """Ambient information a guard may consult while inspecting text.

    ``estimated_cost_usd`` lets a cost gate decide before the model runs;
    ``metadata`` carries anything else a caller wants to thread through (tenant
    id, request class, etc.). All fields are optional -- a guard that needs a
    field it was not given should treat it as absent, never raise.
    """

    stage: Stage
    estimated_cost_usd: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Guard(ABC):
    """A single input/output check.

    Subclasses implement :meth:`inspect` (a coroutine, so LLM-backed guards fit
    the same contract as regex ones) and set :attr:`stage` to the stage they
    belong to. :attr:`mode` decides what happens when the guard triggers and is
    caller-configurable at construction time.
    """

    #: Which stage this guard runs in; subclasses override.
    stage: Stage = Stage.INPUT

    def __init__(self, *, mode: Mode = Mode.BLOCK, name: str | None = None) -> None:
        self.mode = mode
        self.name = name or self.__class__.__name__

    @abstractmethod
    async def inspect(self, text: str, context: GuardContext) -> GuardFinding:
        """Return this guard's finding for ``text``. Must not raise on bad input."""
        raise NotImplementedError

    # -- helpers for subclasses -------------------------------------------
    def _ok(self) -> GuardFinding:
        return GuardFinding(guard=self.name, mode=self.mode, triggered=False)

    def _hit(
        self,
        detail: str,
        *,
        count: int = 0,
        redacted_text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> GuardFinding:
        return GuardFinding(
            guard=self.name,
            mode=self.mode,
            triggered=True,
            detail=detail,
            count=count,
            redacted_text=redacted_text,
            metadata=metadata or {},
        )
