"""Data types for Code Archaeology results."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

SourceType = Literal["git", "markdown", "slack", "email", "mesh"]


@dataclass(frozen=True)
class Citation:
    """A specific evidence reference supporting a claim."""

    source_type: SourceType
    reference: str  # e.g., "commit abc1234", "PR #42", "slack://C123/msg456"
    content_preview: str  # First ~200 chars of the evidence
    timestamp: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TimelineEvent:
    """A single event in the chronological reconstruction."""

    timestamp: datetime
    source_type: SourceType
    summary: str
    citations: list[Citation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CausalClaim:
    """A causal relationship between events, with evidence."""

    cause: str
    effect: str
    confidence: float  # 0.0–1.0
    citations: list[Citation] = field(default_factory=list)
    verified: bool = False  # True if NeuroSymbolicAgent plausibility check passed
    reasoning: str = ""


@dataclass(frozen=True)
class EvolutionSnapshot:
    """How a file/symbol changed at a specific version."""

    version_hash: str
    date: datetime
    diff_summary: str
    reason: str  # Extracted "why" from commit message/PR/linked discussion
    citations: list[Citation] = field(default_factory=list)


@dataclass
class ArchaeologyResult:
    """Complete result from an archaeology query."""

    query: str
    timeline: list[TimelineEvent] = field(default_factory=list)
    causes: list[CausalClaim] = field(default_factory=list)
    evolution: list[EvolutionSnapshot] = field(default_factory=list)
    narrative: str = ""

    @property
    def all_citations(self) -> list[Citation]:
        """Deduplicated list of all citations across timeline, causes, evolution."""
        seen: set[str] = set()
        result: list[Citation] = []
        for event in self.timeline:
            for c in event.citations:
                if c.reference not in seen:
                    seen.add(c.reference)
                    result.append(c)
        for claim in self.causes:
            for c in claim.citations:
                if c.reference not in seen:
                    seen.add(c.reference)
                    result.append(c)
        for snap in self.evolution:
            for c in snap.citations:
                if c.reference not in seen:
                    seen.add(c.reference)
                    result.append(c)
        return result

    def to_markdown(self) -> str:
        """Render result as a markdown report."""
        lines: list[str] = [f"# Code Archaeology: {self.query}", ""]

        if self.narrative:
            lines.extend(["## Summary", self.narrative, ""])

        if self.timeline:
            lines.append("## Timeline")
            for event in self.timeline:
                date_str = event.timestamp.strftime("%Y-%m-%d %H:%M")
                lines.append(f"- **{date_str}** [{event.source_type}]: {event.summary}")
                for c in event.citations:
                    lines.append(f"  - 📎 {c.reference}: {c.content_preview[:80]}")
            lines.append("")

        if self.causes:
            lines.append("## Causal Chain")
            for claim in self.causes:
                verified = "✅" if claim.verified else "⚠️"
                lines.append(
                    f"- {verified} **{claim.cause}** → **{claim.effect}** "
                    f"(confidence: {claim.confidence:.0%})"
                )
                if claim.reasoning:
                    lines.append(f"  > {claim.reasoning}")
                for c in claim.citations:
                    lines.append(f"  - 📎 {c.reference}")
            lines.append("")

        if self.evolution:
            lines.append("## Evolution History")
            for snap in self.evolution:
                date_str = snap.date.strftime("%Y-%m-%d")
                lines.append(f"- **{date_str}** `{snap.version_hash[:8]}`: {snap.diff_summary}")
                if snap.reason:
                    lines.append(f"  > Why: {snap.reason}")
            lines.append("")

        all_cites = self.all_citations
        if all_cites:
            lines.append(f"## Citations ({len(all_cites)} sources)")
            for i, c in enumerate(all_cites, 1):
                lines.append(f"{i}. [{c.source_type}] {c.reference}")

        return "\n".join(lines)


@dataclass
class SourceConfig:
    """Configuration for archaeology data sources."""

    repo_path: str = "."
    include_git: bool = True
    markdown_roots: list[str] = field(default_factory=list)
    slack_bot_token: str | None = None
    slack_channel_ids: list[str] = field(default_factory=list)
    email_imap_server: str | None = None
    email_address: str | None = None
    email_password: str | None = None
    email_folder: str = "INBOX"
    mesh: Any | None = None  # Optional KnowledgeMesh instance
    max_events: int = 200
    min_citations_per_claim: int = 2
