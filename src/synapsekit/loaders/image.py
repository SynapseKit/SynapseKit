"""Image loader that optionally uses vision LLMs to describe images."""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import Any

from ..llm.multimodal import ImageContent, MultimodalMessage
from .base import Document


class ImageLoader:
    """Load images as Document objects.

    Without an LLM, returns a metadata-only document with ``[Image: <path>]``.
    With an LLM, ``async_load``/``aload`` returns a caption suitable for retrieval.
    """

    def __init__(
        self,
        path: str | Path,
        llm: Any = None,
        prompt: str = "Describe this image in detail for retrieval.",
    ) -> None:
        self._path = Path(path)
        self._llm = llm
        self._prompt = prompt

    def load(self) -> list[Document]:
        """Load image synchronously (metadata/placeholder only)."""
        self._validate_file()
        return [self._placeholder_document()]

    async def async_load(self) -> list[Document]:
        """Load image with optional multimodal LLM captioning."""
        self._validate_file()

        metadata = self._base_metadata()
        if self._llm is None:
            return [Document(text=f"[Image: {self._path}]", metadata=metadata)]

        image = ImageContent.from_file(self._path)
        message = MultimodalMessage(text=self._prompt, images=[image])

        provider = getattr(getattr(self._llm, "config", None), "provider", "openai")
        if provider == "anthropic":
            messages = message.to_anthropic_messages()
        else:
            messages = message.to_openai_messages()

        if hasattr(type(self._llm), "generate_with_messages"):
            description = await self._llm.generate_with_messages(messages)
        else:
            # Backward-compatible fallback for very old/custom LLM wrappers.
            description = await self._llm.generate(self._prompt)

        metadata["description_prompt"] = self._prompt
        return [Document(text=str(description).strip(), metadata=metadata)]

    async def aload(self) -> list[Document]:
        """Alias for loader consistency with other loaders."""
        return await self.async_load()

    def _validate_file(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(f"Image file not found: {self._path}")

    def _base_metadata(self) -> dict[str, Any]:
        mime, _ = mimetypes.guess_type(str(self._path))
        if mime is None:
            mime = "image/png"

        return {
            "source": str(self._path),
            "file": str(self._path),
            "source_type": "image",
            "media_type": mime,
            "file_size": os.path.getsize(self._path),
            "loader": "ImageLoader",
        }

    def _placeholder_document(self) -> Document:
        return Document(text=f"[Image: {self._path}]", metadata=self._base_metadata())
