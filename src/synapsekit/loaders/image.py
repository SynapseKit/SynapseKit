"""Image loader that optionally uses vision LLMs to describe images."""

from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import Any

from .base import Document


class ImageLoader:
    """Load images as Document objects.

    Without an LLM, returns a metadata-only document with ``[Image: <path>]``.
    With an LLM (any object with an async ``generate`` method), returns a
    document containing the LLM's description of the image.
    """

    def __init__(
        self,
        path: str | Path,
        llm: Any = None,
        prompt: str = "Describe this image in detail.",
    ) -> None:
        self._path = Path(path)
        self._llm = llm
        self._prompt = prompt

    def load(self) -> list[Document]:
        """Load image synchronously (no LLM description)."""
        if not self._path.exists():
            raise FileNotFoundError(f"Image file not found: {self._path}")

        mime, _ = mimetypes.guess_type(str(self._path))
        if mime is None:
            mime = "image/png"

        file_size = os.path.getsize(self._path)

        metadata = {
            "source": str(self._path),
            "media_type": mime,
            "file_size": file_size,
        }

        if self._llm is None:
            text = f"[Image: {self._path}]"
            return [Document(text=text, metadata=metadata)]

        # With LLM, still return metadata doc synchronously;
        # use async_load for actual description.
        text = f"[Image: {self._path}]"
        return [Document(text=text, metadata=metadata)]

    async def async_load(self) -> list[Document]:
        """Load image with optional LLM description."""
        if not self._path.exists():
            raise FileNotFoundError(f"Image file not found: {self._path}")

        mime, _ = mimetypes.guess_type(str(self._path))
        if mime is None:
            mime = "image/png"

        file_size = os.path.getsize(self._path)

        metadata = {
            "source": str(self._path),
            "media_type": mime,
            "file_size": file_size,
        }

        if self._llm is None:
            text = f"[Image: {self._path}]"
            return [Document(text=text, metadata=metadata)]

        raw = self._path.read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        data_uri = f"data:{mime};base64,{b64}"

        description = await self._llm.generate(self._prompt, image_url=data_uri)
        metadata["description_prompt"] = self._prompt
        return [Document(text=description, metadata=metadata)]
