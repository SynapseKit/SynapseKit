"""Ambient source plugin that tails PowerShell's PSReadLine history file.

Windows has no zsh/bash ``preexec`` hook, so this is a best-effort,
after-the-fact source: it only sees a command once PSReadLine has flushed it
to disk (i.e. after the user pressed Enter), not before execution. True
pre-exec interception would require injecting a ``Set-PSReadLineKeyHandler``
into the user's ``$PROFILE`` — out of scope for this source.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from ..events import AmbientEvent
from .base import AmbientSourcePlugin

logger = logging.getLogger(__name__)

_CANDIDATE_HISTORY_PATHS: tuple[str, ...] = (
    "Microsoft.PowerShell_history.txt",  # PowerShell 7+ (pwsh)
    "ConsoleHost_history.txt",  # Windows PowerShell 5.1
)


def _default_history_path() -> Path | None:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return None
    base = Path(appdata) / "Microsoft" / "Windows" / "PowerShell" / "PSReadLine"
    for name in _CANDIDATE_HISTORY_PATHS:
        candidate = base / name
        if candidate.exists():
            return candidate
    return None


class TerminalSourcePlugin(AmbientSourcePlugin):
    """Tails the PSReadLine history file for newly-run commands."""

    name = "terminal"
    version = "0.1.0"
    description = "Observes recently run PowerShell commands (best-effort, after-the-fact)."

    def __init__(self, history_path: str | Path | None = None) -> None:
        self._configured_path = Path(history_path) if history_path else None
        self._offset: int | None = None
        self._warned_unavailable = False

    async def poll(self) -> list[AmbientEvent]:
        path = self._configured_path or _default_history_path()
        if path is None or not path.exists():
            if not self._warned_unavailable:
                logger.warning("ambient: no PSReadLine history file found, terminal source disabled")
                self._warned_unavailable = True
            return []

        if self._offset is None:
            # Baseline on first poll: don't replay history that predates the
            # daemon starting, only react to commands run from here on.
            self._offset = path.stat().st_size
            return []

        with open(path, encoding="utf-8", errors="replace") as f:
            f.seek(self._offset)
            new_text = f.read()
            self._offset = f.tell()

        lines = [line.strip() for line in new_text.splitlines() if line.strip()]
        now = datetime.now(UTC)
        return [
            AmbientEvent(source=self.name, kind="command", text=line, timestamp=now)
            for line in lines
        ]
