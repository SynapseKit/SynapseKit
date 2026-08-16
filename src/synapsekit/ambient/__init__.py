"""Ambient agent daemon — observes local activity and proactively notifies (MVP slice of #743)."""

from __future__ import annotations

from .daemon import AmbientDaemon, AmbientDaemonConfig
from .events import AmbientEvent, AmbientState
from .privacy import DEFAULT_AMBIENT_IGNORE, load_disabled_sources
from .rules import DEFAULT_MIN_CONFIDENCE, Intervention, evaluate
from .sources import AmbientSourcePlugin, GitSourcePlugin, TerminalSourcePlugin
from .status import DEFAULT_STATUS_PATH, AmbientStatus, read_status, write_status

__all__ = [
    "DEFAULT_AMBIENT_IGNORE",
    "DEFAULT_MIN_CONFIDENCE",
    "DEFAULT_STATUS_PATH",
    "AmbientDaemon",
    "AmbientDaemonConfig",
    "AmbientEvent",
    "AmbientSourcePlugin",
    "AmbientState",
    "AmbientStatus",
    "GitSourcePlugin",
    "Intervention",
    "TerminalSourcePlugin",
    "evaluate",
    "load_disabled_sources",
    "read_status",
    "write_status",
]
