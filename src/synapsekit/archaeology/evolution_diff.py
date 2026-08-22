"""EvolutionDiff — track how code evolved across versions with rationale."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ._terms import extract_terms
from .types import Citation, EvolutionSnapshot

if TYPE_CHECKING:
    from ..llm.base import BaseLLM
    from ..timetravel.evolution_index import EvolutionEntry, EvolutionIndex

logger = logging.getLogger(__name__)


class EvolutionDiff:
    """Tracks how a file or symbol evolved over time, extracting rationale from commits."""

    def __init__(
        self,
        repo_path: str | Path = ".",
        *,
        evolution_index: EvolutionIndex | None = None,
    ) -> None:
        self.repo_path = Path(repo_path).resolve()
        self._evolution_index = evolution_index

    async def trace(
        self,
        file_or_symbol: str,
        *,
        since: str | datetime | None = None,
        until: str | datetime | None = None,
        llm: BaseLLM | None = None,
    ) -> list[EvolutionSnapshot]:
        """Build a chronological evolution trace for a file, symbol, or question."""
        from ..timetravel.evolution_index import EvolutionIndex
        from ..timetravel.git_backend import GitBackend

        if self._evolution_index is not None:
            # A shared, already-built index ignores since/until scoping at the
            # git-log level (it was built once, unscoped) but `query()` still
            # applies since/until filtering in-memory, so results stay correct.
            index = self._evolution_index
            await asyncio.to_thread(index.ensure_built)
        else:
            index = EvolutionIndex(GitBackend(self.repo_path))
            await asyncio.to_thread(index.build, since=since, until=until)

        # Extract search terms from file_or_symbol (case preserved: EvolutionIndex.query
        # matches file paths/symbols case-sensitively).
        terms = extract_terms(file_or_symbol, lower=False)
        matching_entries: list[EvolutionEntry] = []
        for term in terms:
            res = index.query(term, since=since, until=until)
            if res:
                matching_entries.extend(res)

        if not matching_entries:
            # Whole-string fallback only (e.g. a bare file path) — do NOT fall
            # back further to the full unfiltered history: for natural-language
            # queries (the common case via ArchaeologyAgent.explain()) that
            # would silently return every commit in the repo instead of an
            # empty, honest "nothing matched" result.
            matching_entries = index.query(file_or_symbol, since=since, until=until)

        # Deduplicate and sort chronologically
        seen = set()
        entries: list[EvolutionEntry] = []
        for e in sorted(matching_entries, key=lambda x: x.commit.date):
            key = (e.commit.hash, e.file_path, e.symbol)
            if key not in seen:
                seen.add(key)
                entries.append(e)

        snapshots: list[EvolutionSnapshot] = []
        for entry in entries:
            reason = entry.commit.subject
            if entry.commit.body:
                reason += f" — {entry.commit.body.strip()}"

            citation = Citation(
                source_type="git",
                reference=(
                    f"commit {entry.commit.hash[:8]}"
                    + (f" (PR #{entry.pr_number})" if entry.pr_number else "")
                ),
                content_preview=(entry.diff_snippet[:200] if entry.diff_snippet else reason),
                timestamp=entry.commit.date,
                metadata={
                    "author": entry.commit.author,
                    "file_path": entry.file_path,
                    "symbol": entry.symbol,
                    "lines_added": entry.lines_added,
                    "lines_removed": entry.lines_removed,
                },
            )

            diff_summary = (
                f"{entry.change_type} in {entry.file_path}"
                + (f" [{entry.symbol}]" if entry.symbol else "")
                + f" (+{entry.lines_added}/-{entry.lines_removed})"
            )

            snapshots.append(
                EvolutionSnapshot(
                    version_hash=entry.commit.hash,
                    date=entry.commit.date,
                    diff_summary=diff_summary,
                    reason=reason,
                    citations=[citation],
                )
            )

        return snapshots
