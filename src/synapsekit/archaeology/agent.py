"""ArchaeologyAgent — reconstruct why code exists by fusing multi-source evidence."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ..llm._factory import make_llm
from ..llm.base import BaseLLM
from .causal_linker import CausalLinker
from .evolution_diff import EvolutionDiff
from .timeline_reconstructor import TimelineReconstructor
from .types import ArchaeologyResult, SourceConfig

if TYPE_CHECKING:
    from ..symbolic.agent import NeuroSymbolicAgent

logger = logging.getLogger(__name__)

_NARRATIVE_PROMPT = """\
You are a Code Archaeology expert. Based on the following evidence, write a
clear, developer-friendly explanation answering: "{query}"

## Timeline ({n_events} events from {n_sources} sources)
{timeline_summary}

## Causal Chain ({n_claims} causal links identified)
{causal_summary}

## Evolution History ({n_snapshots} versions)
{evolution_summary}

Write a concise but comprehensive narrative that:
1. Explains WHY the code exists in its current form
2. Traces the decision chain that led to this state
3. Cites specific evidence (commits, PRs, messages, notes)
4. Highlights any still-justified vs. potentially outdated decisions
"""


class ArchaeologyAgent:
    """Agent that fuses git log, notes, and external archives to explain WHY code exists.

    Parameters
    ----------
    sources:
        Configuration for data sources (git, markdown, Slack, email, mesh).
    llm:
        BaseLLM instance. If None, one is created from model/api_key.
    model:
        LLM model name (used if llm is None).
    api_key:
        API key for LLM provider (used if llm is None).
    provider:
        Optional LLM provider name.
    causal_engine:
        Optional NeuroSymbolicAgent for causal plausibility checks.
    """

    def __init__(
        self,
        sources: SourceConfig | None = None,
        *,
        llm: BaseLLM | None = None,
        model: str = "gpt-4o-mini",
        api_key: str = "",
        provider: str | None = None,
        causal_engine: NeuroSymbolicAgent | None = None,
    ) -> None:
        self.sources = sources or SourceConfig()

        if llm is not None:
            self.llm: BaseLLM | None = llm
        else:
            try:
                self.llm = make_llm(
                    model,
                    api_key,
                    provider,
                    "You are a code archaeology and history analyst.",
                    0.3,
                    2048,
                )
            except Exception:
                logger.warning(
                    "ArchaeologyAgent could not construct an LLM (model=%r, provider=%r) — "
                    "falling back to timeline-only mode with no causal linking or narrative.",
                    model,
                    provider,
                    exc_info=True,
                )
                self.llm = None

        repo_path = Path(self.sources.repo_path).resolve()
        from ..timetravel.evolution_index import EvolutionIndex
        from ..timetravel.git_backend import GitBackend

        # Shared across timeline + evolution so `explain()` walks git history once.
        evolution_index = EvolutionIndex(GitBackend(repo_path))
        self.timeline_reconstructor = TimelineReconstructor(
            repo_path, evolution_index=evolution_index
        )
        self.evolution_diff = EvolutionDiff(repo_path, evolution_index=evolution_index)
        self.causal_linker: CausalLinker | None = None
        if self.llm is not None:
            self.causal_linker = CausalLinker(
                self.llm,
                verifier=causal_engine,
                min_citations=self.sources.min_citations_per_claim,
            )
        self.causal_engine = causal_engine

    async def explain(self, query: str) -> ArchaeologyResult:
        """Answer 'Why does this code exist?' with a full causal chain and evidence.

        Returns an ArchaeologyResult with:
        - timeline: chronological events from all sources
        - causes: causal claims with evidence and optional verification
        - evolution: how the code changed over time
        - narrative: human-readable markdown summary
        """
        src = self.sources

        # Phase 1: Reconstruct timeline from all sources
        timeline = await self.timeline_reconstructor.reconstruct(
            query,
            include_git=src.include_git,
            slack_bot_token=src.slack_bot_token,
            slack_channel_ids=src.slack_channel_ids or None,
            email_imap_server=src.email_imap_server,
            email_address=src.email_address,
            email_password=src.email_password,
            email_folder=src.email_folder,
            markdown_roots=[Path(r) for r in src.markdown_roots] or None,
            max_events=src.max_events,
        )

        # Phase 2 & 3: Run causal linking and evolution diff concurrently
        causal_task = (
            self.causal_linker.link(timeline, query)
            if self.causal_linker is not None
            else _empty_list()
        )
        evolution_task = self.evolution_diff.trace(query)

        causes, evolution = await asyncio.gather(causal_task, evolution_task)

        # Phase 4: Generate narrative summary
        narrative = ""
        if self.llm is not None:
            narrative = await self._generate_narrative(
                query,
                timeline,
                causes,
                evolution,
            )

        return ArchaeologyResult(
            query=query,
            timeline=timeline,
            causes=causes,
            evolution=evolution,
            narrative=narrative,
        )

    async def _generate_narrative(
        self,
        query: str,
        timeline: list,
        causes: list,
        evolution: list,
    ) -> str:
        """Generate a human-readable narrative from the evidence."""
        if self.llm is None:
            return ""

        source_types = {e.source_type for e in timeline}
        timeline_lines = [
            f"- [{e.timestamp.strftime('%Y-%m-%d')}] ({e.source_type}) {e.summary}"
            for e in timeline[:30]
        ]
        causal_lines = [
            f"- {c.cause} → {c.effect} (confidence: {c.confidence:.0%}, verified: {c.verified})"
            for c in causes
        ]
        evolution_lines = [
            f"- [{s.date.strftime('%Y-%m-%d')}] {s.diff_summary}: {s.reason[:100]}"
            for s in evolution[:20]
        ]

        prompt = _NARRATIVE_PROMPT.format(
            query=query,
            n_events=len(timeline),
            n_sources=len(source_types),
            timeline_summary="\n".join(timeline_lines) or "No events found.",
            n_claims=len(causes),
            causal_summary="\n".join(causal_lines) or "No causal links identified.",
            n_snapshots=len(evolution),
            evolution_summary="\n".join(evolution_lines) or "No evolution data.",
        )

        chunks: list[str] = []
        async for chunk in self.llm.stream(prompt):
            chunks.append(chunk)
        return "".join(chunks)


async def _empty_list() -> list:
    """Async no-op returning an empty list."""
    return []
