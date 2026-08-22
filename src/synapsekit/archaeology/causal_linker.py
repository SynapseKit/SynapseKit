"""CausalLinker — extract and verify causal claims from multi-source evidence."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from .types import CausalClaim, Citation, TimelineEvent

if TYPE_CHECKING:
    from ..llm.base import BaseLLM
    from ..symbolic.agent import NeuroSymbolicAgent

logger = logging.getLogger(__name__)

_CAUSAL_PROMPT = """\
Given the following chronological events related to the query "{query}",
identify causal relationships (what caused what). For each causal link, provide:
1. The cause (what happened first that led to the next event)
2. The effect (what resulted from the cause)
3. Your confidence level (0.0 to 1.0)
4. Brief reasoning

Events:
{events}

Return your analysis as a numbered list of causal links in this exact format:
CAUSE: <description>
EFFECT: <description>
CONFIDENCE: <0.0-1.0>
REASONING: <explanation>
---
"""


class CausalLinker:
    """Extracts causal relationships from timeline events and optionally verifies them."""

    def __init__(
        self,
        llm: BaseLLM,
        *,
        verifier: NeuroSymbolicAgent | None = None,
        min_citations: int = 2,
    ) -> None:
        self.llm = llm
        self.verifier = verifier
        self.min_citations = min_citations

    async def link(
        self,
        events: list[TimelineEvent],
        query: str,
    ) -> list[CausalClaim]:
        """Extract causal claims from timeline events and verify them."""
        if not events:
            return []

        # Format events for LLM
        event_strs: list[str] = []
        for i, e in enumerate(events[:50], 1):  # Cap for context length
            cite_refs = ", ".join(c.reference for c in e.citations)
            event_strs.append(
                f"{i}. [{e.timestamp.strftime('%Y-%m-%d')}] "
                f"({e.source_type}) {e.summary} [refs: {cite_refs}]"
            )

        prompt = _CAUSAL_PROMPT.format(
            query=query,
            events="\n".join(event_strs),
        )

        response_chunks: list[str] = []
        async for chunk in self.llm.stream(prompt):
            response_chunks.append(chunk)
        response = "".join(response_chunks)

        claims = self._parse_claims(response, events)

        # Filter claims with insufficient evidence
        claims = [c for c in claims if len(c.citations) >= self.min_citations]

        # Verify with NeuroSymbolicAgent if available
        if self.verifier is not None:
            claims = await self._verify_claims(claims)

        return claims

    def _parse_claims(
        self,
        response: str,
        events: list[TimelineEvent],
    ) -> list[CausalClaim]:
        """Parse LLM response into CausalClaim objects."""
        claims: list[CausalClaim] = []
        blocks = response.split("---")

        all_citations: list[Citation] = []
        for e in events:
            all_citations.extend(e.citations)

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            cause = effect = reasoning = ""
            confidence = 0.5

            for line in block.split("\n"):
                line = line.strip()
                if line.upper().startswith("CAUSE:"):
                    cause = line[6:].strip()
                elif line.upper().startswith("EFFECT:"):
                    effect = line[7:].strip()
                elif line.upper().startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line[11:].strip())
                    except ValueError:
                        confidence = 0.5
                elif line.upper().startswith("REASONING:"):
                    reasoning = line[10:].strip()

            if cause and effect:
                # Find matching citations from events
                matched_citations = self._match_citations(
                    cause + " " + effect,
                    all_citations,
                )
                claims.append(
                    CausalClaim(
                        cause=cause,
                        effect=effect,
                        confidence=confidence,
                        citations=matched_citations,
                        verified=False,
                        reasoning=reasoning,
                    )
                )

        return claims

    def _match_citations(
        self,
        text: str,
        all_citations: list[Citation],
    ) -> list[Citation]:
        """Find citations whose content is relevant to the claim text."""
        text_lower = text.lower()
        terms = [w for w in text_lower.split() if len(w) > 3]

        scored: list[tuple[int, Citation]] = []
        for c in all_citations:
            preview_lower = c.content_preview.lower()
            ref_lower = c.reference.lower()
            score = sum(1 for t in terms if t in preview_lower or t in ref_lower)
            if score > 0:
                scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:5]]

    async def _verify_claims(
        self,
        claims: list[CausalClaim],
    ) -> list[CausalClaim]:
        """Best-effort claim verification via NeuroSymbolicAgent.

        NeuroSymbolicAgent.solve() formalizes its input into an SMT/Prolog/
        MiniZinc/Sympy constraint program and checks it with a solver — it was
        built for problems with decidable formal structure, not prose
        plausibility judgments. Free-text causal claims have no reliable
        formalization, so `verified` here is a best-effort signal, not a
        formal proof; treat it as informational rather than authoritative.
        """
        if self.verifier is None:
            return claims

        logger.info(
            "CausalLinker verifying %d claim(s) via NeuroSymbolicAgent — "
            "verdicts are best-effort, not formal proofs, since causal claims "
            "are prose rather than a decidable constraint problem.",
            len(claims),
        )

        results = await asyncio.gather(
            *(self._verify_claim(claim) for claim in claims),
            return_exceptions=True,
        )

        verified_claims: list[CausalClaim] = []
        for claim, result in zip(claims, results, strict=True):
            if isinstance(result, BaseException):
                logger.warning(
                    "Verification failed for claim: %s -> %s (%s)",
                    claim.cause,
                    claim.effect,
                    result,
                )
                verified_claims.append(claim)
            else:
                verified_claims.append(result)

        return verified_claims

    async def _verify_claim(self, claim: CausalClaim) -> CausalClaim:
        assert self.verifier is not None
        evidence = [c.content_preview for c in claim.citations]
        problem = (
            f"Verify this causal claim: '{claim.cause}' caused '{claim.effect}'. "
            f"Evidence: {'; '.join(evidence[:3])}"
        )
        result = await self.verifier.solve(problem)
        return CausalClaim(
            cause=claim.cause,
            effect=claim.effect,
            confidence=claim.confidence,
            citations=claim.citations,
            verified=bool(getattr(result, "verified", False)),
            reasoning=claim.reasoning,
        )
