from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from synapsekit.llm.base import BaseLLM
from synapsekit.twin.style_profile import LearnedPatterns

logger = logging.getLogger(__name__)


@dataclass
class VoiceMatchResult:
    score: float
    ngram_overlap: float
    vocabulary_match: float
    structure_match: float
    details: dict[str, Any] = field(default_factory=dict)


class VoiceMatcher:
    """Evaluates how well a draft matches human writing style reference samples."""

    def __init__(self, llm: BaseLLM | None = None) -> None:
        self.llm = llm

    async def evaluate(
        self,
        candidate: str,
        reference_samples: Sequence[str],
        patterns: LearnedPatterns | None = None,
    ) -> VoiceMatchResult:
        if patterns is None:
            patterns = LearnedPatterns()

        ngram = self._compute_ngram_overlap(candidate, reference_samples)
        vocab = self._compute_vocabulary_match(candidate, patterns)
        struct = self._compute_structure_match(candidate, patterns)

        # Composite weighted score: 40% ngram, 30% vocab, 30% structure
        heuristic_score = (0.4 * ngram) + (0.3 * vocab) + (0.3 * struct)

        # LLM judge evaluation if available
        llm_score = heuristic_score
        if self.llm is not None and reference_samples:
            try:
                llm_score = await self._evaluate_with_llm(candidate, reference_samples)
            except Exception as err:
                logger.debug("LLM voice match evaluation failed: %s", err)

        final_score = round((heuristic_score + llm_score) / 2.0, 4)

        return VoiceMatchResult(
            score=final_score,
            ngram_overlap=round(ngram, 4),
            vocabulary_match=round(vocab, 4),
            structure_match=round(struct, 4),
            details={
                "heuristic_score": round(heuristic_score, 4),
                "llm_score": round(llm_score, 4),
            },
        )

    def _compute_ngram_overlap(self, candidate: str, references: Sequence[str]) -> float:
        if not candidate or not references:
            return 0.5

        def get_bigrams(text: str) -> set[tuple[str, str]]:
            words = text.lower().split()
            return {(words[i], words[i + 1]) for i in range(len(words) - 1)}

        cand_bigrams = get_bigrams(candidate)
        if not cand_bigrams:
            return 0.5

        ref_bigrams: set[tuple[str, str]] = set()
        for ref in references:
            ref_bigrams.update(get_bigrams(ref))

        if not ref_bigrams:
            return 0.5

        overlap = cand_bigrams.intersection(ref_bigrams)
        return len(overlap) / float(len(cand_bigrams))

    def _compute_vocabulary_match(self, candidate: str, patterns: LearnedPatterns) -> float:
        if not candidate or not patterns.vocabulary:
            return 1.0

        cand_words = set(candidate.lower().split())
        matched = 0
        total = len(patterns.vocabulary)

        for _avoid, preferred in patterns.vocabulary.items():
            if preferred.lower() in cand_words:
                matched += 1

        return float(matched) / float(total) if total > 0 else 1.0

    def _compute_structure_match(self, candidate: str, patterns: LearnedPatterns) -> float:
        if not candidate:
            return 0.5

        has_bullets = "- " in candidate or "* " in candidate
        if patterns.structure == "bulleted":
            return 1.0 if has_bullets else 0.4
        elif patterns.structure == "prose":
            return 0.4 if has_bullets else 1.0
        return 0.8

    async def _evaluate_with_llm(self, candidate: str, references: Sequence[str]) -> float:
        if self.llm is None:
            return 0.8

        prompt = (
            "Compare this candidate draft against the human author's sample writings:\n\n"
            "Reference samples:\n"
            + "\n---\n".join(references[:3])
            + f"\n\nCandidate draft:\n{candidate}\n\n"
            f"Rate how well the candidate matches the tone, style, and vocabulary (0.0 to 1.0). Output ONLY a single floating-point number."
        )

        response_chunks: list[str] = []
        async for chunk in self.llm.stream(prompt):
            response_chunks.append(chunk)

        full_text = "".join(response_chunks).strip()
        try:
            val = float(full_text)
            return max(0.0, min(1.0, val))
        except ValueError:
            return 0.8
