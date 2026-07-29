"""Narrative generator for code evolution and diff timelines."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .evolution_index import EvolutionEntry

if TYPE_CHECKING:
    from ..llm.base import BaseLLM


class DiffNarrativeGenerator:
    """Generates human-readable evolutionary summaries from diff timelines."""

    def __init__(self, llm: BaseLLM | None = None) -> None:
        self.llm = llm

    async def generate(
        self,
        entries: list[EvolutionEntry],
        query: str,
        llm: BaseLLM | None = None,
    ) -> str:
        """Generate evolution narrative for a query using LLM or heuristic fallback."""
        active_llm = llm or self.llm
        if active_llm is not None:
            try:
                return await self._llm_narrative(entries, query, active_llm)
            except Exception:
                pass
        return self._heuristic_narrative(entries, query)

    def _heuristic_narrative(self, entries: list[EvolutionEntry], query: str) -> str:
        """Generate structured markdown summary without an LLM."""
        if not entries:
            return f"No evolution history found for query: '{query}'."

        sorted_entries = sorted(entries, key=lambda e: e.commit.date)
        first_date = sorted_entries[0].commit.date.strftime("%Y-%m-%d")
        last_date = sorted_entries[-1].commit.date.strftime("%Y-%m-%d")

        lines = [
            f"# Code Evolution Summary for '{query}'",
            f"**Timeframe**: {first_date} to {last_date} ({len(sorted_entries)} changes)",
            "",
            "### Evolution Timeline",
        ]

        for entry in sorted_entries:
            c = entry.commit
            date_str = c.date.strftime("%Y-%m-%d")
            pr_str = f" (#{entry.pr_number})" if entry.pr_number else ""
            sym_str = f" [`{entry.symbol}`]" if entry.symbol else ""
            lines.append(
                f"- **{date_str}** `[{c.hash[:7]}]`{pr_str}{sym_str}: {c.subject} (by {c.author})"
            )
            if entry.diff_snippet:
                snippet_preview = entry.diff_snippet.strip().split("\n")[0]
                lines.append(f"  > `{snippet_preview}`")

        lines.extend(
            [
                "",
                "### Key Changes & Rationale",
                f"- Total modifications: {sum(e.lines_added + e.lines_removed for e in sorted_entries)} lines changed.",
                f"- Primary files affected: {', '.join(sorted({e.file_path for e in sorted_entries}))}.",
            ]
        )

        return "\n".join(lines)

    async def _llm_narrative(
        self,
        entries: list[EvolutionEntry],
        query: str,
        llm: BaseLLM,
    ) -> str:
        """Generate narrative using LLM analysis."""
        timeline_str = []
        for e in entries[:20]:  # Cap entries for context length
            pr_str = f" (PR #{e.pr_number})" if e.pr_number else ""
            timeline_str.append(
                f"- Commit {e.commit.hash[:8]} on {e.commit.date.strftime('%Y-%m-%d')}{pr_str}: "
                f"{e.commit.subject}\n  Diff excerpt: {e.diff_snippet[:200]}"
            )

        prompt = (
            f"Analyze the evolution of code relevant to query: '{query}'.\n\n"
            f"## Change History:\n" + "\n".join(timeline_str) + "\n\n"
            "Provide a concise, developer-friendly narrative explaining:\n"
            "1. How the code evolved over time\n"
            "2. Why the changes were introduced\n"
            "3. Any architectural shift or design rationale"
        )

        response_chunks = []
        async for chunk in llm.stream(prompt):
            response_chunks.append(chunk)

        return "".join(response_chunks)
