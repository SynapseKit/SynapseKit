"""AuditLog — immutable, append-only compliance log."""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class AuditEntry:
    """Single immutable audit record."""

    entry_id: str
    timestamp: str
    user: str
    model: str
    input_text: str
    output_text: str
    cost_usd: float | None = None
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AuditLog:
    """Immutable, append-only compliance log.

    Supports three backends:

    - ``"memory"`` — in-process list (default, no persistence)
    - ``"sqlite"`` — stdlib ``sqlite3``, indexed on user/model/timestamp
    - ``"jsonl"`` — append-only file, loads existing entries on startup

    No delete or update API — immutable by design.
    Thread-safe via ``threading.Lock``.

    Example::

        log = AuditLog(backend="sqlite", path="audit.db")
        entry = log.record(
            model="gpt-4o-mini",
            input_text="What is 2+2?",
            output_text="4",
            cost_usd=0.001,
            latency_ms=120.0,
            user="alice",
        )
        results = log.query(user="alice", limit=10)
    """

    def __init__(self, backend: str = "memory", path: str | None = None) -> None:
        if backend not in ("memory", "sqlite", "jsonl"):
            raise ValueError(f"backend must be 'memory', 'sqlite', or 'jsonl', got {backend!r}")
        self._backend = backend
        self._path = path
        self._lock = threading.Lock()
        self._entries: list[AuditEntry] = []
        self._conn: Any = None

        if backend == "sqlite":
            self._init_sqlite()
        elif backend == "jsonl":
            self._init_jsonl()

    # ------------------------------------------------------------------ #
    # Backend initialisation
    # ------------------------------------------------------------------ #

    def _init_sqlite(self) -> None:
        import sqlite3

        self._conn = sqlite3.connect(self._path or "audit_log.db", check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                entry_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                user TEXT NOT NULL,
                model TEXT NOT NULL,
                input_text TEXT NOT NULL,
                output_text TEXT NOT NULL,
                cost_usd REAL,
                latency_ms REAL,
                metadata TEXT
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log (user)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_model ON audit_log (model)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log (timestamp)")
        self._conn.commit()

    def _init_jsonl(self) -> None:
        path = Path(self._path or "audit_log.jsonl")
        if path.exists():
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        self._entries.append(AuditEntry(**data))

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #

    def record(
        self,
        model: str,
        input_text: str,
        output_text: str,
        cost_usd: float | None = None,
        latency_ms: float | None = None,
        user: str = "anonymous",
        metadata: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Append a new audit entry. Returns the created entry."""
        entry = AuditEntry(
            entry_id=uuid.uuid4().hex,
            timestamp=datetime.now(timezone.utc).isoformat(),
            user=user,
            model=model,
            input_text=input_text,
            output_text=output_text,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )

        with self._lock:
            if self._backend == "memory":
                self._entries.append(entry)
            elif self._backend == "sqlite":
                self._write_sqlite(entry)
            elif self._backend == "jsonl":
                self._entries.append(entry)
                self._append_jsonl(entry)

        return entry

    def _write_sqlite(self, entry: AuditEntry) -> None:
        self._conn.execute(
            """
            INSERT INTO audit_log
                (entry_id, timestamp, user, model, input_text, output_text, cost_usd, latency_ms, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.entry_id,
                entry.timestamp,
                entry.user,
                entry.model,
                entry.input_text,
                entry.output_text,
                entry.cost_usd,
                entry.latency_ms,
                json.dumps(entry.metadata),
            ),
        )
        self._conn.commit()

    def _append_jsonl(self, entry: AuditEntry) -> None:
        path = Path(self._path or "audit_log.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    # ------------------------------------------------------------------ #
    # Querying
    # ------------------------------------------------------------------ #

    def query(
        self,
        user: str | None = None,
        model: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int | None = None,
    ) -> list[AuditEntry]:
        """Query audit entries with optional filters."""
        with self._lock:
            if self._backend == "sqlite":
                return self._query_sqlite(user, model, since, until, limit)
            return self._query_memory(user, model, since, until, limit)

    def _query_memory(
        self,
        user: str | None,
        model: str | None,
        since: str | None,
        until: str | None,
        limit: int | None,
    ) -> list[AuditEntry]:
        results = self._entries
        if user:
            results = [e for e in results if e.user == user]
        if model:
            results = [e for e in results if e.model == model]
        if since:
            results = [e for e in results if e.timestamp >= since]
        if until:
            results = [e for e in results if e.timestamp <= until]
        if limit:
            results = results[:limit]
        return results

    def _query_sqlite(
        self,
        user: str | None,
        model: str | None,
        since: str | None,
        until: str | None,
        limit: int | None,
    ) -> list[AuditEntry]:
        sql = "SELECT entry_id, timestamp, user, model, input_text, output_text, cost_usd, latency_ms, metadata FROM audit_log WHERE 1=1"
        params: list[Any] = []
        if user:
            sql += " AND user = ?"
            params.append(user)
        if model:
            sql += " AND model = ?"
            params.append(model)
        if since:
            sql += " AND timestamp >= ?"
            params.append(since)
        if until:
            sql += " AND timestamp <= ?"
            params.append(until)
        sql += " ORDER BY timestamp"
        if limit:
            sql += " LIMIT ?"
            params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [
            AuditEntry(
                entry_id=r[0],
                timestamp=r[1],
                user=r[2],
                model=r[3],
                input_text=r[4],
                output_text=r[5],
                cost_usd=r[6],
                latency_ms=r[7],
                metadata=json.loads(r[8]) if r[8] else {},
            )
            for r in rows
        ]

    # ------------------------------------------------------------------ #
    # Dunder
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        with self._lock:
            if self._backend == "sqlite":
                row = self._conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()
                return int(row[0])
            return len(self._entries)
