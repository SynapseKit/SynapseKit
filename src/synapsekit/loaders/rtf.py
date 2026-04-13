from __future__ import annotations

import asyncio
import os

from .base import Document


class RTFLoader:
    """Load an RTF file and extract plain text using striprtf.

    Requires: pip install synapsekit[rtf]
    """

    def __init__(self, path: str, encoding: str = "utf-8") -> None:
        self._path = path
        self._encoding = encoding

    def load(self) -> list[Document]:
        if not os.path.exists(self._path):
            raise FileNotFoundError(f"RTF file not found: {self._path}")
        try:
            from striprtf.striprtf import rtf_to_text
        except ImportError:
            raise ImportError("striprtf required: pip install synapsekit[rtf]") from None

        with open(self._path, encoding=self._encoding, errors="ignore") as f:
            raw = f.read()

        if not raw:
            return [Document(text="", metadata={"source": self._path})]

        # striprtf provides best-effort conversion; malformed RTF may not fully parse
        text = rtf_to_text(raw) or ""
        text = text.strip()
        return [Document(text=text, metadata={"source": self._path})]

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)
