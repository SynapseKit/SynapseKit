from __future__ import annotations

import asyncio
import mimetypes
import os
from pathlib import Path

from .base import Document


class PDFLoader:
    """Load a PDF file, one Document per page."""

    def __init__(self, path: str) -> None:
        self._path = path

    async def aload(self) -> list[Document]:
        return await asyncio.to_thread(self.load)

    def load(self) -> list[Document]:
        if not os.path.exists(self._path):
            raise FileNotFoundError(f"PDF file not found: {self._path}")
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("pypdf required: pip install synapsekit[pdf]") from None

        reader = PdfReader(self._path)
        docs = []
        media_type, _ = mimetypes.guess_type(self._path)
        source_name = Path(self._path).name
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            page_number = i + 1
            docs.append(
                Document(
                    text=text,
                    metadata={
                        "source": self._path,
                        "file": self._path,
                        "source_type": "pdf",
                        "media_type": media_type or "application/pdf",
                        "loader": "PDFLoader",
                        "chunk_type": "page",
                        "page": page_number,
                        "locator": f"{source_name} page {page_number}",
                    },
                )
            )
        return docs
