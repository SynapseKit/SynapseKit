"""Local-only trace and Dream Mode state storage."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from ..audit import AuditRecord, EventKind
from .types import DreamRunResult, Lesson, MeshConsolidation, StaleMemory, TraceWindow


class DreamStateStore:
    """SQLite journal for traces, memory reads, and completed dream reports.

    The store is intentionally independent from the memory patch store and the
    mesh index.  Dream Mode can therefore be enabled or removed without
    changing either product's persistence format.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def append_traces(self, records: Iterable[AuditRecord]) -> int:
        """Persist trace records idempotently and return newly stored count."""

        rows = [
            (
                record.event_id,
                record.run_id,
                record.timestamp.astimezone(UTC).isoformat(),
                json.dumps(record.to_dict(), sort_keys=True),
            )
            for record in records
        ]
        if not rows:
            return 0
        with self._lock, self._conn:
            before = self._conn.total_changes
            self._conn.executemany(
                """
                INSERT OR IGNORE INTO traces(event_id, run_id, timestamp, record_json)
                VALUES (?, ?, ?, ?)
                """,
                rows,
            )
            inserted = self._conn.total_changes - before
        return int(inserted)

    def records(self, window: TraceWindow) -> list[AuditRecord]:
        """Return local traces in timestamp order for ``window``."""

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT record_json FROM traces
                WHERE run_id IN (SELECT run_id FROM traces WHERE timestamp >= ? AND timestamp <= ?)
                ORDER BY timestamp, event_id
                """,
                (window.start.astimezone(UTC).isoformat(), window.end.astimezone(UTC).isoformat()),
            ).fetchall()
        return [AuditRecord.from_dict(json.loads(str(row["record_json"]))) for row in rows]

    def update_memory_reads(self, records: Iterable[AuditRecord]) -> int:
        """Record ``MEMORY_READ`` paths found in imported traces."""

        values: list[tuple[str, str]] = []
        for record in records:
            if record.kind != EventKind.MEMORY_READ.value:
                continue
            path = record.payload.get("path")
            if isinstance(path, str) and path.strip():
                values.append((path, record.timestamp.astimezone(UTC).isoformat()))
        if not values:
            return 0
        with self._lock, self._conn:
            self._conn.executemany(
                """
                INSERT INTO memory_reads(path, last_read_at) VALUES (?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    last_read_at = CASE
                        WHEN excluded.last_read_at > memory_reads.last_read_at
                        THEN excluded.last_read_at
                        ELSE memory_reads.last_read_at
                    END
                """,
                values,
            )
        return len(values)

    def stale_memories(
        self,
        paths: Iterable[str | Path],
        *,
        now: datetime,
        stale_after_days: int,
    ) -> list[StaleMemory]:
        """Return old memory files without deleting or mutating them."""

        cutoff = now.astimezone(UTC).timestamp() - stale_after_days * 86400
        path_list = [Path(path).expanduser() for path in paths]
        if not path_list:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT path, last_read_at FROM memory_reads WHERE path IN ({})".format(
                    ",".join("?" for _ in path_list)
                ),
                [str(path) for path in path_list],
            ).fetchall()
        last_reads = {str(row["path"]): str(row["last_read_at"]) for row in rows}
        stale: list[StaleMemory] = []
        for path in path_list:
            try:
                stat = path.stat()
            except OSError:
                continue
            modified = datetime.fromtimestamp(stat.st_mtime, UTC)
            read_text = last_reads.get(str(path))
            try:
                read_at = datetime.fromisoformat(read_text) if read_text else None
            except ValueError:
                read_at = None
            reference = max(modified, read_at or datetime.fromtimestamp(0, UTC))
            if reference.timestamp() >= cutoff:
                continue
            age_days = max(0, int((now.astimezone(UTC) - reference).total_seconds() // 86400))
            stale.append(
                StaleMemory(
                    path=str(path),
                    last_read_at=read_text,
                    last_modified_at=modified.isoformat(),
                    age_days=age_days,
                    reason="not read or modified within the configured retention window",
                )
            )
        return sorted(stale, key=lambda item: (-item.age_days, item.path))

    def save_run(self, result: DreamRunResult) -> None:
        payload = json.dumps(result.to_dict(), sort_keys=True)
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO dream_runs(run_id, started_at, status, report_json)
                VALUES (?, ?, ?, ?)
                """,
                (result.run_id, result.started_at, result.status, payload),
            )

    def last_run(self) -> DreamRunResult | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT report_json FROM dream_runs ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        data = json.loads(str(row["report_json"]))
        data["lessons"] = [Lesson(**lesson) for lesson in data.get("lessons", [])]
        data["mesh_consolidations"] = [
            MeshConsolidation(**item) for item in data.get("mesh_consolidations", [])
        ]
        data["stale_memories"] = [StaleMemory(**item) for item in data.get("stale_memories", [])]
        return DreamRunResult(**data)

    def _ensure_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    event_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_traces_timestamp ON traces(timestamp);
                CREATE TABLE IF NOT EXISTS memory_reads (
                    path TEXT PRIMARY KEY,
                    last_read_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS dream_runs (
                    run_id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    report_json TEXT NOT NULL
                );
                """
            )
            self._conn.commit()
