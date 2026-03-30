from __future__ import annotations

import os
import xml.etree.ElementTree as ET

from .base import Document


class XMLLoader:
    """Load an XML file and extract text content from elements."""

    def __init__(
        self,
        path: str,
        tags: list[str] | None = None,
        encoding: str = "utf-8",
    ) -> None:
        self._path = path
        self._tags = tags
        self._encoding = encoding

    def load(self) -> list[Document]:
        if not os.path.exists(self._path):
            raise FileNotFoundError(f"XML file not found: {self._path}")

        root = ET.parse(self._path).getroot()

        if self._tags:
            text_parts = [
                text
                for tag in self._tags
                for elem in root.findall(f".//{tag}")
                if (text := " ".join(elem.itertext()).strip())
            ]
            text = "\n".join(text_parts)
        else:
            text = " ".join(root.itertext()).strip()

        return [Document(text=text, metadata={"source": self._path})]
