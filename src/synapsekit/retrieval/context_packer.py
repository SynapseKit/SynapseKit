"""Long-context chunk packing with dedup and ordering strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from ..loaders.base import Document
from .token_counting import TokenCounter


@dataclass
class PackedChunk:
    text: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    token_count: int = 0


class ContextPacker:
    """Pack retrieved chunks into a strict token budget."""

    def __init__(
        self,
        max_tokens: int = 180_000,
        strategy: str = "relevance",
        dedup_threshold: float = 0.92,
        ordering: str = "lost-in-middle",
        token_counter: TokenCounter | None = None,
        model: str | None = None,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if strategy not in {"relevance", "recency", "diversity"}:
            raise ValueError("strategy must be one of: relevance, recency, diversity")
        if ordering not in {"as-is", "lost-in-middle"}:
            raise ValueError("ordering must be one of: as-is, lost-in-middle")

        self.max_tokens = max_tokens
        self.strategy = strategy
        self.dedup_threshold = dedup_threshold
        self.ordering = ordering
        self._counter = token_counter or TokenCounter(model=model, backend="auto")

    def pack(self, chunks: list[Any], query: str | None = None) -> list[dict[str, Any]]:
        """Return packed chunks as dicts with text/score/metadata/token_count."""
        normalized = self._normalize(chunks, query=query)
        if not normalized:
            return []

        deduped = self._deduplicate(normalized)
        ranked = self._rank(deduped, query=query)
        budgeted = self._apply_token_budget(ranked)
        ordered = self._order(budgeted)
        return [
            {
                "text": c.text,
                "score": c.score,
                "metadata": c.metadata,
                "token_count": c.token_count,
            }
            for c in ordered
        ]

    def pack_texts(self, chunks: list[Any], query: str | None = None) -> list[str]:
        return [c["text"] for c in self.pack(chunks, query=query)]

    def _normalize(self, chunks: list[Any], query: str | None = None) -> list[PackedChunk]:
        normalized: list[PackedChunk] = []
        q_terms = self._token_set(query or "")

        for idx, item in enumerate(chunks):
            if isinstance(item, str):
                text = item
                score = 0.0
                metadata = {"rank": idx}
            elif isinstance(item, Document):
                text = item.text
                metadata = dict(item.metadata or {})
                score = self._as_float(metadata.get("score", metadata.get("relevance_score", 0.0)))
            elif isinstance(item, dict):
                text = str(item.get("text", ""))
                metadata = dict(item.get("metadata") or {})
                score = self._as_float(
                    item.get(
                        "score",
                        item.get(
                            "relevance_score",
                            item.get("cross_encoder_score", metadata.get("score", 0.0)),
                        ),
                    )
                )
            else:
                text = str(item)
                score = 0.0
                metadata = {"rank": idx}

            text = text.strip()
            if not text:
                continue

            if score == 0.0 and q_terms:
                score = self._lexical_overlap_score(text, q_terms)

            tokens = self._counter.count_cached(text)
            normalized.append(
                PackedChunk(
                    text=text,
                    score=score,
                    metadata=metadata,
                    token_count=tokens,
                )
            )

        return normalized

    def _deduplicate(self, chunks: list[PackedChunk]) -> list[PackedChunk]:
        if not chunks:
            return []

        # Keep the strongest chunk among near-duplicates.
        if self.strategy == "recency":

            def _recency_key(c: PackedChunk) -> float:
                value = (
                    c.metadata.get("timestamp")
                    or c.metadata.get("created_at")
                    or c.metadata.get("updated_at")
                    or c.metadata.get("date")
                    or 0
                )
                return self._as_float(value)

            ordered = sorted(chunks, key=_recency_key, reverse=True)
        else:
            ordered = sorted(chunks, key=lambda c: c.score, reverse=True)

        kept: list[PackedChunk] = []
        for chunk in ordered:
            is_dup = any(
                self._similarity(chunk.text, prev.text) >= self.dedup_threshold for prev in kept
            )
            if not is_dup:
                kept.append(chunk)
        return kept

    def _rank(self, chunks: list[PackedChunk], query: str | None = None) -> list[PackedChunk]:
        if self.strategy == "relevance":
            return sorted(chunks, key=lambda c: c.score, reverse=True)

        if self.strategy == "recency":

            def _recency_key(c: PackedChunk) -> float:
                value = (
                    c.metadata.get("timestamp")
                    or c.metadata.get("created_at")
                    or c.metadata.get("updated_at")
                    or c.metadata.get("date")
                    or 0
                )
                return self._as_float(value)

            return sorted(chunks, key=_recency_key, reverse=True)

        # diversity
        if not chunks:
            return []

        candidates = sorted(chunks, key=lambda c: c.score, reverse=True)
        selected: list[PackedChunk] = [candidates[0]]
        remaining = candidates[1:]

        while remaining:
            best_idx = -1
            best_value = float("-inf")
            for idx, item in enumerate(remaining):
                max_sim = max(self._similarity(item.text, chosen.text) for chosen in selected)
                value = 0.6 * item.score + 0.4 * (1.0 - max_sim)
                if value > best_value:
                    best_value = value
                    best_idx = idx
            selected.append(remaining.pop(best_idx))

        return selected

    def _apply_token_budget(self, ranked: list[PackedChunk]) -> list[PackedChunk]:
        packed: list[PackedChunk] = []
        used = 0
        for chunk in ranked:
            if chunk.token_count <= 0:
                continue
            if used + chunk.token_count > self.max_tokens:
                continue
            packed.append(chunk)
            used += chunk.token_count
        return packed

    def _order(self, packed: list[PackedChunk]) -> list[PackedChunk]:
        if self.ordering != "lost-in-middle" or len(packed) <= 2:
            return packed

        # Lost-in-middle mitigation: strongest chunks placed at start/end.
        ordered: list[PackedChunk | None] = [None] * len(packed)
        left = 0
        right = len(packed) - 1

        for idx, chunk in enumerate(packed):
            if idx % 2 == 0:
                ordered[left] = chunk
                left += 1
            else:
                ordered[right] = chunk
                right -= 1

        return [c for c in ordered if c is not None]

    @staticmethod
    def _token_set(text: str) -> set[str]:
        return {t for t in text.lower().split() if t}

    def _lexical_overlap_score(self, text: str, query_terms: set[str]) -> float:
        if not query_terms:
            return 0.0
        terms = self._token_set(text)
        if not terms:
            return 0.0
        return len(terms & query_terms) / len(query_terms)

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    @staticmethod
    def _as_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
