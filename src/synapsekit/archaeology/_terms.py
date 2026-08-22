"""Shared query-term extraction/matching for archaeology sources."""

from __future__ import annotations

_STRIP_CHARS = "?,.'\"`"


def extract_terms(query: str, *, min_len: int = 3, lower: bool = True) -> list[str]:
    """Split a query into search terms, dropping short/punctuation-only tokens.

    `lower=False` preserves case for callers matching against case-sensitive
    identifiers (e.g. Python symbol names via `EvolutionIndex.query`).
    """
    terms = []
    for t in query.split():
        stripped = t.strip(_STRIP_CHARS)
        if lower:
            stripped = stripped.lower()
        if len(stripped) >= min_len:
            terms.append(stripped)
    return terms


def matches(text: str, terms: list[str]) -> bool:
    """Return True if any term appears in text (case-insensitive)."""
    text_lower = text.lower()
    return any(term in text_lower for term in terms)
