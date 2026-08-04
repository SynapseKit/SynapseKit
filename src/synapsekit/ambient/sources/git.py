"""Ambient source plugin that polls local git status."""

from __future__ import annotations

import asyncio
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from ..events import AmbientEvent
from .base import AmbientSourcePlugin

logger = logging.getLogger(__name__)


class GitSourcePlugin(AmbientSourcePlugin):
    """Polls ``git status --porcelain --branch`` for working-tree changes."""

    name = "git"
    version = "0.1.0"
    description = "Observes local git branch and working-tree status."

    def __init__(self, repo_root: str | Path = ".") -> None:
        self.repo_root = Path(repo_root)
        self._last_raw: str | None = None
        self._unavailable = False

    async def poll(self) -> list[AmbientEvent]:
        if self._unavailable:
            return []

        raw = await asyncio.to_thread(self._run_status)
        if raw is None:
            self._unavailable = True
            return []
        if raw == self._last_raw:
            return []
        self._last_raw = raw

        lines = raw.splitlines()
        branch = None
        if lines and lines[0].startswith("##"):
            header = lines[0][2:].strip()
            branch = header.split("...")[0].strip() or None
            lines = lines[1:]
        dirty_files = [line[3:].strip() for line in lines if line.strip()]

        return [
            AmbientEvent(
                source=self.name,
                kind="git_status",
                text=f"{len(dirty_files)} file(s) changed on {branch or 'unknown branch'}",
                timestamp=datetime.now(UTC),
                metadata={
                    "dirty": bool(dirty_files),
                    "dirty_files": dirty_files,
                    "branch": branch,
                },
            )
        ]

    def _run_status(self) -> str | None:
        # ponytail: no retry/backoff after first failure (not-a-repo, git
        # missing from PATH); add if this proves noisy in practice.
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain", "--branch"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            logger.warning("ambient: git source unavailable, disabling", exc_info=True)
            return None
        if result.returncode != 0:
            logger.warning("ambient: not a git repo at %s, disabling git source", self.repo_root)
            return None
        return result.stdout
