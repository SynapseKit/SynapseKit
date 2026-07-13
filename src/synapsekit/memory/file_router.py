"""Route facts to the appropriate memory file based on content category."""

from __future__ import annotations

import re
from pathlib import Path

from .living_types import MemoryFileCategory


class MemoryFileRouter:
    """Determine which memory file a new fact should be written to.

    Routes based on content analysis and configured path mappings.
    Falls back to the primary memory file when no specific route matches.

    Parameters
    ----------
    path_map:
        Explicit mapping of category to file path, e.g.
        ``{"user": "./memory/user_prefs.md"}``.
    primary_path:
        Default file path when no category-specific route matches.
    allow_new_files:
        When True, the router may suggest paths for files that don't
        yet exist on disk.  Defaults to False.
    """

    # Keyword patterns that suggest content category
    _CATEGORY_SIGNALS: dict[MemoryFileCategory, list[str]] = {
        "user": [
            r"\bprefer(?:s|ence|red)\b",
            r"\balways\s+(?:use|choose|want)\b",
            r"\bmy\s+(?:style|workflow|setup)\b",
            r"\bI\s+(?:like|dislike|want|need)\b",
        ],
        "feedback": [
            r"\bcorrect(?:ion|ed|ing)\b",
            r"\bfix(?:ed)?\b",
            r"\bwrong\b",
            r"\bmistake\b",
            r"\bbetter\s+(?:to|if)\b",
        ],
        "project": [
            r"\barchitecture\b",
            r"\bstack\b",
            r"\bdependenc(?:y|ies)\b",
            r"\bconvention\b",
            r"\bcodebase\b",
            r"\brepository\b",
        ],
    }

    def __init__(
        self,
        path_map: dict[MemoryFileCategory, str] | None = None,
        *,
        primary_path: str = "./CLAUDE.md",
        allow_new_files: bool = False,
    ) -> None:
        self._primary = primary_path
        self._allow_new = allow_new_files
        self._path_map: dict[MemoryFileCategory, str] = path_map or {}
        self._compiled: dict[MemoryFileCategory, list[re.Pattern[str]]] = {}
        for cat, patterns in self._CATEGORY_SIGNALS.items():
            self._compiled[cat] = [re.compile(p, re.IGNORECASE) for p in patterns]

    def categorize(self, content: str) -> MemoryFileCategory:
        """Determine the category of a piece of content.

        Scores each category by counting regex hits and returns the
        best match, falling back to ``'general'`` when nothing matches.
        """
        scores: dict[MemoryFileCategory, int] = {
            "user": 0,
            "feedback": 0,
            "project": 0,
            "general": 0,
        }

        for category, patterns in self._compiled.items():
            for pattern in patterns:
                hits = len(pattern.findall(content))
                scores[category] += hits

        best_category = max(scores, key=scores.get)  # type: ignore[arg-type]
        if scores[best_category] == 0:
            return "general"
        return best_category

    def resolve_target_path(
        self,
        category: MemoryFileCategory,
        managed_paths: list[str],
    ) -> str:
        """Determine which file path a fact should be routed to.

        Resolution order:
        1. Explicit ``path_map`` entry for the category.
        2. A managed file whose basename starts with the category name.
        3. The primary memory file as fallback.
        """
        # Check explicit path map first
        if category in self._path_map:
            candidate = self._path_map[category]
            if candidate in managed_paths or self._allow_new:
                return candidate

        # Try to find a managed file whose name suggests this category
        category_prefix = f"{category}_"
        for managed in managed_paths:
            basename = Path(managed).stem.lower()
            if basename.startswith(category_prefix) or basename == category:
                return managed

        # Fall back to primary
        return self._primary
