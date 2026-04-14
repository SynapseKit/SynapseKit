from __future__ import annotations

import asyncio
import os
import re

from .base import Document


class LaTeXLoader:
    """Load and convert a LaTeX (.tex) file to plain text.

    Uses regex-based heuristics to strip commands, environments, and math
    blocks. Complex macros or deeply nested structures may not be handled
    perfectly. Section titles are captured in metadata when present.
    """

    def __init__(self, path: str, encoding: str = "utf-8") -> None:
        self._path = path
        self._encoding = encoding

    def load(self) -> list[Document]:
        if not os.path.exists(self._path):
            raise FileNotFoundError(f"File not found: {self._path}")

        with open(self._path, encoding=self._encoding) as f:
            raw = f.read()

        sections = self._extract_sections(raw)
        cleaned = self._strip_latex(raw)

        metadata: dict = {"source": self._path}
        if sections:
            metadata["title"] = sections[0]
            metadata["sections"] = sections

        return [Document(text=cleaned, metadata=metadata)]

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)

    def _extract_sections(self, text: str) -> list[str]:
        return re.findall(r"\\(?:sub)*section\*?\{([^}]*)\}", text)

    def _strip_latex(self, text: str) -> str:
        text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)
        text = re.sub(r"\$[^$]*?\$", "", text)
        text = re.sub(r"\\begin\{[^}]*\}.*?\\end\{[^}]*\}", "", text, flags=re.DOTALL)
        text = re.sub(r"%[^\n]*", "", text)
        text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})*", "", text)
        text = re.sub(r"[{}\\]", "", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
