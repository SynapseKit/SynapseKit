from __future__ import annotations

import asyncio
from collections.abc import Iterable

from .base import Document


def _flatten_sections(sections: Iterable[object]) -> list[str]:
    parts: list[str] = []
    for section in sections:
        title = getattr(section, "title", "")
        text = getattr(section, "text", "")
        if title:
            parts.append(str(title))
        if text:
            parts.append(str(text))
        nested = getattr(section, "sections", [])
        if nested:
            parts.extend(_flatten_sections(nested))
    return parts


class WikipediaLoader:
    """Load Wikipedia articles by title using the wikipedia-api library.

    Args:
        query: Article title to look up (exact Wikipedia page title).
        language: Wikipedia language edition, e.g. ``"en"``, ``"de"``.
        max_results: Maximum number of pages to return. Currently fetches
            the single page matching *query*; increase to load additional
            articles by passing a list of titles.
    """

    def __init__(
        self,
        query: str,
        language: str = "en",
        max_results: int = 1,
    ) -> None:
        self._query = query
        self._language = language
        self._max_results = max_results

    def _build_doc(self, page: object) -> Document:
        summary = getattr(page, "summary", "") or ""
        sections = _flatten_sections(getattr(page, "sections", []))
        text = "\n\n".join([summary, *sections]).strip()

        categories = getattr(page, "categories", {}) or {}
        metadata = {
            "title": getattr(page, "title", None),
            "url": getattr(page, "fullurl", None),
            "categories": list(categories.keys()),
            "language": self._language,
            "source": getattr(page, "fullurl", None),
        }
        return Document(text=text, metadata=metadata)

    def load(self) -> list[Document]:
        try:
            import wikipediaapi
        except ImportError:
            raise ImportError("wikipedia-api required: pip install synapsekit[wikipedia]") from None

        wiki = wikipediaapi.Wikipedia(
            user_agent="SynapseKit/1.0",
            language=self._language,
        )

        titles = [t.strip() for t in self._query.split("|") if t.strip()]
        if not titles:
            return []

        docs: list[Document] = []
        for title in titles[: self._max_results]:
            page = wiki.page(title)
            if hasattr(page, "exists") and not page.exists():
                continue
            docs.append(self._build_doc(page))

        return docs

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)
