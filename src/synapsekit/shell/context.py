"""Context providers used by the shell planner."""

from __future__ import annotations

import asyncio
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Protocol

from .adapters import detect_shell
from .types import ShellContext, ShellKind


class AmbientContextProvider(Protocol):
    async def snapshot(self) -> dict[str, Any]: ...


class NullAmbientContext:
    """No-op provider used when the optional Ambient Agent is unavailable."""

    async def snapshot(self) -> dict[str, Any]:
        return {}


class JsonAmbientContext:
    """Read a bounded status document produced by an Ambient Agent daemon."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()

    async def snapshot(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._read)

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8")[:128_000])
        except (FileNotFoundError, OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}


def _git_value(cwd: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            check=False,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()[:4_096]


class ContextCollector:
    """Collect local context without leaking an unrestricted environment."""

    _SAFE_ENV_KEYS = (
        "CI",
        "GITHUB_ACTIONS",
        "GITHUB_REF_NAME",
        "VIRTUAL_ENV",
        "CONDA_DEFAULT_ENV",
        "TERM",
    )

    def __init__(
        self,
        *,
        cwd: str | Path | None = None,
        shell: ShellKind | str | None = None,
        mesh: Any | None = None,
        ambient: AmbientContextProvider | None = None,
        history: Any | None = None,
    ) -> None:
        self.cwd = Path(cwd or Path.cwd()).expanduser().resolve()
        self.shell = (
            shell
            if isinstance(shell, ShellKind)
            else (detect_shell() if shell is None else ShellKind.parse(str(shell)))
        )
        self.mesh = mesh
        self.ambient = ambient or NullAmbientContext()
        self.history = history

    async def collect(self, query: str = "") -> ShellContext:
        branch, status = await asyncio.gather(
            asyncio.to_thread(_git_value, self.cwd, "branch", "--show-current"),
            asyncio.to_thread(_git_value, self.cwd, "status", "--short", "--branch"),
        )
        mesh_hits: tuple[dict[str, Any], ...] = ()
        mesh = self.mesh
        if mesh is not None and query:
            mesh_hits = await asyncio.to_thread(self._query_mesh, query)
        ambient, history = await asyncio.gather(
            self.ambient.snapshot(),
            self._history_search(query),
        )
        environment = {key: os.environ[key] for key in self._SAFE_ENV_KEYS if os.environ.get(key)}
        return ShellContext(
            cwd=str(self.cwd),
            platform=platform.platform(),
            shell=self.shell,
            git_branch=branch,
            git_status=status,
            mesh_hits=mesh_hits,
            ambient=ambient,
            history=history,
            environment=environment,
        )

    def _query_mesh(self, query: str) -> tuple[dict[str, Any], ...]:
        mesh = self.mesh
        if mesh is None:
            return ()
        try:
            result = mesh.query_sync(query, top_k=5)
        except Exception:
            return ()
        return tuple(
            {
                "path": hit.path,
                "line_start": hit.line_start,
                "line_end": hit.line_end,
                "text": " ".join(hit.text.split())[:500],
                "score": hit.score,
            }
            for hit in result.hits
        )

    async def _history_search(self, query: str) -> tuple[dict[str, Any], ...]:
        if self.history is None or not query:
            return ()
        try:
            return tuple(await self.history.search(query, limit=5))
        except Exception:
            return ()
