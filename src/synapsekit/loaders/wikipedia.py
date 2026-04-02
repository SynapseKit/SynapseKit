from __future__ import annotations

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
    """Load Wikipedia articles by title or search query."""

    def __init__(self, query: str, language: str = "en", max_results: int = 1) -> None:
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
            language=self._language,
            extract_format=getattr(wikipediaapi, "ExtractFormat", None),
        )

        pages: list[object] = []
        search_fn = getattr(wiki, "search", None)
        if callable(search_fn):
            try:
                results = search_fn(self._query, results=self._max_results)
            except TypeError:
                results = search_fn(self._query)
            for title in list(results)[: self._max_results]:
                pages.append(wiki.page(title))
        else:
            pages.append(wiki.page(self._query))

        docs: list[Document] = []
        for page in pages:
            if hasattr(page, "exists") and not page.exists():
                continue
            docs.append(self._build_doc(page))

        return docs
