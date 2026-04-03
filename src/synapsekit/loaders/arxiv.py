from __future__ import annotations

import asyncio
import re
import tempfile
from collections.abc import Iterable
from datetime import datetime

from .base import Document
from .pdf import PDFLoader

_ARXIV_ID_RE = re.compile(r"^\s*\d{4}\.\d{4,5}(v\d+)?\s*$")


def _is_arxiv_id(value: str) -> bool:
    value = value.strip()
    if _ARXIV_ID_RE.match(value):
        return True
    if " " in value:
        return False
    return "/" in value


def _author_names(authors: Iterable[object]) -> list[str]:
    names: list[str] = []
    for author in authors:
        name = getattr(author, "name", None)
        if name:
            names.append(name)
        else:
            names.append(str(author))
    return names


def _format_date(value: object) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    return None


class ArXivLoader:
    """Load papers from arXiv by ID or search query."""

    def __init__(self, query: str, max_results: int = 1) -> None:
        self._query = query
        self._max_results = max_results

    def load(self) -> list[Document]:
        try:
            import arxiv
        except ImportError:
            raise ImportError("arxiv required: pip install synapsekit[arxiv]") from None

        if _is_arxiv_id(self._query):
            search = arxiv.Search(id_list=[self._query.strip()], max_results=self._max_results)
        else:
            search = arxiv.Search(query=self._query, max_results=self._max_results)

        client = arxiv.Client()
        docs: list[Document] = []
        for result in client.results(search):
            with tempfile.TemporaryDirectory() as tmpdir:
                pdf_path = result.download_pdf(dirpath=tmpdir)
                pages = PDFLoader(pdf_path).load()

            text = "\n\n".join(page.text for page in pages)
            metadata = {
                "source": getattr(result, "entry_id", None),
                "arxiv_id": getattr(result, "get_short_id", lambda: None)(),
                "title": getattr(result, "title", None),
                "authors": _author_names(getattr(result, "authors", [])),
                "abstract": getattr(result, "summary", None),
                "published": _format_date(getattr(result, "published", None)),
                "pdf_url": getattr(result, "pdf_url", None),
            }
            docs.append(Document(text=text, metadata=metadata))

        return docs

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)
