"""EvolutionDiff — track how code evolved across versions with rationale."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .types import Citation, EvolutionSnapshot

if TYPE_CHECKING:
    from ..llm.base import BaseLLM
    from ..timetravel.evolution_index import EvolutionEntry

logger = logging.getLogger(__name__)


class EvolutionDiff:
    """Tracks how a file or symbol evolved over time, extracting rationale from commits."""

    def __init__(self, repo_path: str | Path = ".") -> None:
        self.repo_path = Path(repo_path).resolve()

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

        backend = GitBackend(self.repo_path)
        index = EvolutionIndex(backend)
        all_entries = await asyncio.to_thread(index.build, since=since, until=until)

        # Extract search terms from file_or_symbol
        terms = [t.strip("?,.'\"`") for t in file_or_symbol.split() if len(t) > 2]
        matching_entries: list[EvolutionEntry] = []
        for term in terms:
            res = index.query(term, since=since, until=until)
            if res:
                matching_entries.extend(res)

        if not matching_entries:
            matching_entries = index.query(file_or_symbol, since=since, until=until)
        if not matching_entries:
            matching_entries = all_entries

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
                content_preview=(
                    entry.diff_snippet[:200]
                    if entry.diff_snippet
                    else reason
                ),
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
