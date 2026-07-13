"""JSONL-backed patch storage for Living Memory."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .living_types import MemoryPatch, OccurrenceRecord, PatchStatus

_log = logging.getLogger(__name__)


class PatchStore:
    """Append-only JSONL store for memory patches with query helpers.

    Patches are persisted as one JSON object per line.  Updates to an
    existing patch (e.g. status change) are appended as new lines;
    queries always return the latest version for each ``patch_id``.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._patches: list[MemoryPatch] = []
        if self._path.exists():
            self._load()

    def save(self, patch: MemoryPatch) -> MemoryPatch:
        """Sign and append a new patch to the store."""
        if not patch.signature:
            patch.sign()
        self._patches.append(patch)
        self._append_line(patch.to_dict())
        return patch

    def update(self, patch: MemoryPatch) -> None:
        """Re-sign and persist the updated patch by appending a new line."""
        patch.sign()
        self._append_line(patch.to_dict())

    def get(self, patch_id: str) -> MemoryPatch | None:
        """Retrieve the latest version of a patch by its ID."""
        for patch in reversed(self._patches):
            if patch.patch_id == patch_id:
                return patch
        return None

    def list_by_status(
        self,
        status: PatchStatus | None = None,
        *,
        limit: int | None = None,
    ) -> list[MemoryPatch]:
        """Return patches filtered by status, newest first, deduplicated by ID."""
        result = [
            p for p in reversed(self._patches)
            if status is None or p.status == status
        ]
        # Deduplicate by patch_id (keep latest)
        seen: set[str] = set()
        deduped: list[MemoryPatch] = []
        for p in result:
            if p.patch_id not in seen:
                seen.add(p.patch_id)
                deduped.append(p)
        if limit is not None:
            return deduped[:limit]
        return deduped

    def pending_patches(self) -> list[MemoryPatch]:
        """Shorthand for listing all patches with status ``'pending'``."""
        return self.list_by_status("pending")

    def count(self, status: PatchStatus | None = None) -> int:
        """Count patches, optionally filtered by status."""
        return len(self.list_by_status(status))

    def _append_line(self, data: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(data, sort_keys=True, default=str) + "\n")

    def _load(self) -> None:
        with self._path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    self._patches.append(MemoryPatch.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError) as exc:
                    _log.warning("Skipping corrupt patch at line %d: %s", line_num, exc)


class OccurrenceTracker:
    """Track fact occurrences across sessions to avoid single-observation noise.

    Facts must be observed at least ``min_occurrences`` times (across
    different sessions) before they are eligible for patch proposals.
    State is persisted as a JSON file.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else None
        self._records: dict[str, OccurrenceRecord] = {}
        if self._path and self._path.exists():
            self._load()

    def record_occurrence(
        self,
        fact_key: str,
        session_id: str,
        evidence: str = "",
    ) -> OccurrenceRecord:
        """Record one observation of *fact_key* in the given session."""
        if fact_key not in self._records:
            self._records[fact_key] = OccurrenceRecord(fact_key=fact_key)
        rec = self._records[fact_key]
        rec.increment(session_id, evidence)
        self._persist()
        return rec

    def get_mature_facts(self, min_occurrences: int = 3) -> list[OccurrenceRecord]:
        """Return facts that have been observed at least *min_occurrences* times."""
        return [
            r for r in self._records.values()
            if r.count >= min_occurrences
        ]

    def has_reached_threshold(self, fact_key: str, threshold: int = 3) -> bool:
        """Check whether a fact has met the occurrence threshold."""
        rec = self._records.get(fact_key)
        return rec is not None and rec.count >= threshold

    def get_count(self, fact_key: str) -> int:
        """Return the current occurrence count for a fact, or 0 if unseen."""
        rec = self._records.get(fact_key)
        return rec.count if rec is not None else 0

    def remove(self, fact_key: str) -> None:
        """Delete tracking data for a fact."""
        self._records.pop(fact_key, None)
        self._persist()

    def _persist(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: vars(v) for k, v in self._records.items()}
        self._path.write_text(
            json.dumps(data, sort_keys=True, default=str, indent=2),
            encoding="utf-8",
        )

    def _load(self) -> None:
        if self._path is None:
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            for key, val in raw.items():
                self._records[key] = OccurrenceRecord(**val)
        except (json.JSONDecodeError, TypeError):
            _log.warning("Could not parse occurrence tracker at %s", self._path)
