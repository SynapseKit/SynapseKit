from __future__ import annotations

from .base import Document


class PDFLoader:
    """Load a PDF file, one Document per page."""

    def __init__(self, path: str) -> None:
        self._path = path

    def load(self) -> list[Document]:
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("pypdf required: pip install synapsekit[pdf]") from None

        reader = PdfReader(self._path)
        docs = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            docs.append(Document(text=text, metadata={"source": self._path, "page": i}))
        return docs
