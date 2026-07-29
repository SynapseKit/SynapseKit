"""TimeTravelAgent — reason across code evolution, git history, and design rationale."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ..llm._factory import make_llm
from ..llm.base import BaseLLM
from .drift_detector import DriftCandidate, DriftDetector
from .evolution_index import EvolutionEntry, EvolutionIndex
from .git_backend import AsOf, GitBackend, _parse_datetime
from .narrative import DiffNarrativeGenerator

if TYPE_CHECKING:
    from ..memory.living_memory import LivingMemory
    from ..retrieval.world_model import WorldModelRAG


class TimeTravelAgent:
    """Agent that reasons across code evolution, git history, and memory docs over time.

    Parameters
    ----------
    repo:
        Path to the git repository root.
    llm:
        BaseLLM instance. Created automatically from model/api_key if None.
    model:
        Default LLM model name.
    api_key:
        API key for LLM provider.
    provider:
        Optional LLM provider name.
    world_model:
        Optional WorldModelRAG instance for graph retrieval.
    memory:
        Optional LivingMemory instance for historical patch retrieval.
    """

    def __init__(
        self,
        repo: str | Path = ".",
        *,
        llm: BaseLLM | None = None,
        model: str = "gpt-4o-mini",
        api_key: str = "",
        provider: str | None = None,
        world_model: WorldModelRAG | None = None,
        memory: LivingMemory | None = None,
    ) -> None:
        self.repo_path = Path(repo).resolve()
        self.backend = GitBackend(self.repo_path)
        self.index = EvolutionIndex(self.backend)
        self.drift_detector = DriftDetector(self.backend, self.index)

        self.llm: BaseLLM | None
        if llm is not None:
            self.llm = llm
        else:
            try:
                self.llm = make_llm(
                    model,
                    api_key,
                    provider,
                    "You are a code evolution analyst assistant.",
                    0.2,
                    1024,
                )
            except Exception:
                self.llm = None

        self.narrative_generator = DiffNarrativeGenerator(self.llm)
        self.world_model = world_model
        self.memory = memory

    def as_of(self, date: str | datetime) -> AsOf:
        """Return a point-in-time context for querying historical codebase state."""
        dt = _parse_datetime(date) or datetime.now(UTC)
        commit = self.backend.find_commit_at(dt)
        return AsOf(agent=self, date=dt, commit=commit)

    async def timeline(
        self,
        file_or_symbol: str,
        since: str | datetime | None = None,
        until: str | datetime | None = None,
    ) -> list[EvolutionEntry]:
        """Fetch chronological timeline of evolution entries for a file or symbol."""
        return await asyncio.to_thread(
            self._timeline_sync, file_or_symbol, since=since, until=until
        )

    def _timeline_sync(
        self,
        file_or_symbol: str,
        since: str | datetime | None = None,
        until: str | datetime | None = None,
    ) -> list[EvolutionEntry]:
        """Blocking timeline computation (runs git subprocess pipeline)."""
        return self.index.timeline(file_or_symbol, since=since, until=until)

    async def detect_drift(
        self,
        symbol: str | None = None,
        min_age_days: int = 0,
        as_of_date: datetime | None = None,
    ) -> list[DriftCandidate]:
        """Detect abstractions whose original justification or caller count has drifted."""
        return await asyncio.to_thread(
            self._detect_drift_sync,
            symbol=symbol,
            min_age_days=min_age_days,
            as_of_date=as_of_date,
        )

    def _detect_drift_sync(
        self,
        symbol: str | None = None,
        min_age_days: int = 0,
        as_of_date: datetime | None = None,
    ) -> list[DriftCandidate]:
        """Blocking drift detection (runs git subprocess pipeline)."""
        return self.drift_detector.detect(
            min_age_days=min_age_days,
            symbol=symbol,
            as_of_date=as_of_date,
        )

    async def query(self, question: str) -> str:
        """Answer a question about code evolution, design changes, and history."""
        # Offload the blocking git/index pipeline off the event loop.
        deduped, memory_notes = await asyncio.to_thread(self._query_collect_sync, question)

        # Incorporate world model bitemporal graph if attached (async IO)
        if self.world_model is not None:
            try:
                wm_res = await self.world_model.query(question)
                if wm_res and hasattr(wm_res, "answer"):
                    memory_notes.append(f"World Model Graph Context: {wm_res.answer}")
            except Exception:
                pass

        narrative = await self.narrative_generator.generate(deduped[:30], question, self.llm)
        if memory_notes:
            narrative += "\n\n### Historical Memory & Knowledge Graph Context\n" + "\n".join(
                memory_notes
            )

        return narrative

    def _query_collect_sync(
        self, question: str
    ) -> tuple[list[EvolutionEntry], list[str]]:
        """Blocking part of query(): build/query the evolution index off the event loop."""
        # Check if question specifies a class or file symbol
        entries = self.index.build()

        # Extract target search terms from question
        terms = [t for t in question.split() if len(t) > 2]
        matching_entries: list[EvolutionEntry] = []

        for term in terms:
            clean_term = term.strip("?,.'\"`")
            res = self.index.query(clean_term)
            if res:
                matching_entries.extend(res)

        if not matching_entries:
            matching_entries = entries

        # Deduplicate
        seen = set()
        deduped = []
        for e in matching_entries:
            key = (e.commit.hash, e.file_path, e.symbol)
            if key not in seen:
                seen.add(key)
                deduped.append(e)

        # Incorporate memory file patches if memory instance attached
        memory_notes: list[str] = []
        if self.memory is not None:
            try:
                patches = self.memory.patch_history()
                for p in patches:
                    if any(
                        t.lower() in p.unified_diff.lower() or t.lower() in p.rationale.lower()
                        for t in terms
                    ):
                        memory_notes.append(f"Memory Patch ({p.file_path}): {p.rationale}")
            except Exception:
                pass

        return deduped, memory_notes

    async def _query_as_of(self, question: str, date: datetime, commit: str) -> str:
        """Query codebase state as of a specific historical date and commit."""
        # Offload the blocking git/index pipeline off the event loop.
        lines, entries = await asyncio.to_thread(
            self._query_as_of_sync, question, date, commit
        )

        narrative = await self.narrative_generator.generate(entries[:20], question, self.llm)

        return "\n\n".join(lines) + "\n\n" + narrative

    def _query_as_of_sync(
        self, question: str, date: datetime, commit: str
    ) -> tuple[list[str], list[EvolutionEntry]]:
        """Blocking part of _query_as_of(): read files/index at a commit off the event loop."""
        # Query files and contents at that commit
        files = self.backend.list_files(commit)
        date_str = date.strftime("%Y-%m-%d")

        relevant_files = [
            f for f in files if any(t.lower() in f.lower() for t in question.split() if len(t) > 3)
        ]

        lines = [
            f"# Codebase State As Of {date_str} (`{commit[:8]}`)",
            f"**Query**: {question}",
            f"**Tracked Files at Commit**: {len(files)} files",
        ]

        if relevant_files:
            lines.append("\n### Relevant Files at Date:")
            for rf in relevant_files[:10]:
                content = self.backend.file_at(commit, rf)
                first_lines = content.splitlines()[:5]
                preview = "\n  ".join(first_lines)
                lines.append(f"- **`{rf}`**:\n  {preview}")

        # Generate timeline up to commit
        entries = self.index.build(until=date)

        return lines, entries
