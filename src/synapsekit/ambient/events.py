"""Event and state types shared by ambient source plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class AmbientEvent:
    """A single observation reported by an ambient source plugin."""

    source: str
    kind: str
    text: str
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AmbientState:
    """Latest known signal from each source, updated in place each tick."""

    git_dirty: bool = False
    dirty_files: tuple[str, ...] = ()
    branch: str | None = None
    head: str | None = None

    def apply_git_status(self, event: AmbientEvent) -> None:
        self.git_dirty = bool(event.metadata.get("dirty", False))
        self.dirty_files = tuple(event.metadata.get("dirty_files", ()))
        self.branch = event.metadata.get("branch")
        self.head = event.metadata.get("head")
