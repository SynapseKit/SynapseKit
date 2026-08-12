"""Local, redacted semantic-ish shell history backed by SQLite."""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

_WORD_RE = re.compile(r"[a-z0-9_./:-]+", re.IGNORECASE)


def _redact(value: str) -> str:
    """Remove common secret-shaped values before local persistence."""

    value = re.sub(r"(?i)(api[_-]?key|token|secret|password)=([^\s]+)", r"\1=<redacted>", value)
    value = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer <redacted>", value)
    return value[:16_384]


class ShellHistory:
    """Persist shell interactions locally and search by meaningful terms."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(
            path or Path.home() / ".synapsekit" / "shell" / "history.sqlite3"
        ).expanduser()
        self._ready = False
        self._ready_lock = asyncio.Lock()

    async def _ensure_ready(self) -> None:
        if self._ready:
            return
        async with self._ready_lock:
            if not self._ready:
                await asyncio.to_thread(self._initialize)
                self._ready = True

    def _initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS shell_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL NOT NULL,
                    cwd TEXT NOT NULL,
                    input_text TEXT NOT NULL,
                    commands_json TEXT NOT NULL,
                    ok INTEGER NOT NULL
                )"""
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS shell_history_created ON shell_history(created_at DESC)"
            )

    async def record(
        self,
        *,
        cwd: str,
        input_text: str,
        commands: list[str],
        ok: bool,
    ) -> None:
        await self._ensure_ready()
        await asyncio.to_thread(self._record_sync, cwd, input_text, commands, ok)

    def _record_sync(self, cwd: str, input_text: str, commands: list[str], ok: bool) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "INSERT INTO shell_history(created_at,cwd,input_text,commands_json,ok) VALUES(?,?,?,?,?)",
                (
                    time.time(),
                    cwd,
                    _redact(input_text),
                    json.dumps([_redact(c) for c in commands]),
                    int(ok),
                ),
            )

    async def search(self, query: str, *, limit: int = 20) -> list[dict[str, Any]]:
        await self._ensure_ready()
        return await asyncio.to_thread(self._search_sync, query, limit)

    def _search_sync(self, query: str, limit: int) -> list[dict[str, Any]]:
        terms = [term.casefold() for term in _WORD_RE.findall(query) if len(term) > 1]
        with sqlite3.connect(self.path) as connection:
            if terms:
                clauses = " OR ".join("input_text LIKE ?" for _ in terms)
                params: list[Any] = [f"%{term}%" for term in terms]
                rows = connection.execute(
                    f"SELECT created_at,cwd,input_text,commands_json,ok FROM shell_history WHERE {clauses} ORDER BY created_at DESC LIMIT ?",
                    [*params, limit],
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT created_at,cwd,input_text,commands_json,ok FROM shell_history ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [
            {
                "created_at": row[0],
                "cwd": row[1],
                "input": row[2],
                "commands": json.loads(row[3]),
                "ok": bool(row[4]),
            }
            for row in rows
        ]

    async def recent(self, *, limit: int = 20) -> list[dict[str, Any]]:
        return await self.search("", limit=limit)
