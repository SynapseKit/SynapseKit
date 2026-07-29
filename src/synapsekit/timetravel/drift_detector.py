"""Drift detector for identifying stale abstractions whose original justification no longer holds."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime

from .evolution_index import EvolutionEntry, EvolutionIndex
from .git_backend import CommitInfo, GitBackend


@dataclass(frozen=True)
class DriftCandidate:
    """Represents a symbol or abstraction flagged as architectural drift candidate."""

    symbol: str
    file_path: str
    original_rationale: str
    current_usage_count: int
    original_usage_count: int
    first_introduced: datetime
    last_modified: datetime
    confidence: float
    recommendation: str


class DriftDetector:
    """Detects codebase abstractions whose original design constraints or callers have drifted."""

    def __init__(self, backend: GitBackend, index: EvolutionIndex | None = None) -> None:
        self.backend = backend
        self.index = index or EvolutionIndex(backend)

    def _count_usages(self, symbol: str, commit: str | None = None) -> int:
        """Count callers/references of a symbol in the repo at a commit or HEAD."""
        files = self.backend.list_files(commit)
        count = 0
        pattern = re.compile(rf"\b{re.escape(symbol)}\b")

        for f in files:
            if not f.endswith(".py"):
                continue
            content = self.backend.file_at(commit or "HEAD", f)
            matches = len(pattern.findall(content))
            if matches > 0:
                count += matches

        return count

    def _extract_rationale(self, symbol: str, commit: CommitInfo) -> str:
        """Extract design rationale from the commit subject and body where symbol was added."""
        rationale = f"{commit.subject}. {commit.body}".strip()
        if not rationale:
            rationale = f"Introduced in commit {commit.hash[:8]} by {commit.author}."
        return rationale

    def detect(
        self,
        paths: list[str] | None = None,
        min_age_days: int = 0,
        symbol: str | None = None,
        as_of_date: datetime | None = None,
    ) -> list[DriftCandidate]:
        """Detect drift candidates across indexed symbols."""
        entries = self.index.build(paths=paths, until=as_of_date)
        if not entries:
            return []

        # Group entries by symbol
        symbols_map: dict[str, list[EvolutionEntry]] = {}
        for entry in entries:
            if entry.symbol:
                if symbol and entry.symbol != symbol:
                    continue
                symbols_map.setdefault(entry.symbol, []).append(entry)

        now = as_of_date or datetime.now(UTC)
        candidates: list[DriftCandidate] = []

        for sym, sym_entries in symbols_map.items():
            sym_entries_sorted = sorted(sym_entries, key=lambda e: e.commit.date)
            first_entry = sym_entries_sorted[0]
            last_entry = sym_entries_sorted[-1]

            first_commit = first_entry.commit
            last_commit = last_entry.commit

            age_days = (now - first_commit.date).days
            if age_days < min_age_days:
                continue

            first_commit_hash = first_commit.hash
            original_usage = self._count_usages(sym, first_commit_hash)
            current_usage = self._count_usages(sym, "HEAD")

            rationale = self._extract_rationale(sym, first_commit)

            # Heuristic calculation for drift
            # 1. Single caller or zero callers remaining
            # 2. Original usage was higher than current usage
            # 3. Mention of temporary/deprecated/workaround in rationale
            drift_signals = 0.0

            if current_usage <= 1:
                drift_signals += 0.4
            elif current_usage < original_usage:
                drift_signals += 0.3

            if any(
                w in rationale.lower()
                for w in ["temporary", "workaround", "interim", "deprecated", "todo", "routing"]
            ):
                drift_signals += 0.3

            if age_days > 180:
                drift_signals += 0.2

            confidence = min(0.95, drift_signals)

            if confidence >= 0.4 or (symbol and sym == symbol):
                rec = (
                    f"Symbol '{sym}' was added with rationale '{rationale[:100]}...'. "
                    f"Current usage count is {current_usage} (down from {original_usage}). "
                    "Consider simplifying or retiring this abstraction."
                )
                candidates.append(
                    DriftCandidate(
                        symbol=sym,
                        file_path=first_entry.file_path,
                        original_rationale=rationale,
                        current_usage_count=current_usage,
                        original_usage_count=original_usage,
                        first_introduced=first_commit.date,
                        last_modified=last_commit.date,
                        confidence=confidence,
                        recommendation=rec,
                    )
                )

        return sorted(candidates, key=lambda c: c.confidence, reverse=True)
