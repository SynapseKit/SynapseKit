"""Multimodal content types for LLM providers."""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ImageContent:
    """Image content that can be converted to provider-specific formats."""

    data: str  # base64-encoded image data
    media_type: str  # e.g. "image/png", "image/jpeg"
    source_type: str = "base64"  # "base64", "url", "file"
    url: str | None = None  # original URL if from_url

    @classmethod
    def from_file(cls, path: str | Path) -> ImageContent:
        """Create ImageContent from a file path."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        mime, _ = mimetypes.guess_type(str(p))
        if mime is None:
            mime = "image/png"
        raw = p.read_bytes()
        encoded = base64.b64encode(raw).decode("ascii")
        return cls(data=encoded, media_type=mime, source_type="file")

    @classmethod
    def from_url(cls, url: str, media_type: str = "image/png") -> ImageContent:
        """Create ImageContent from a URL (no download, stores URL)."""
        return cls(data="", media_type=media_type, source_type="url", url=url)

    @classmethod
    def from_base64(cls, data: str, media_type: str = "image/png") -> ImageContent:
        """Create ImageContent from a base64-encoded string."""
        return cls(data=data, media_type=media_type, source_type="base64")

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI image_url content block."""
        if self.source_type == "url" and self.url:
            return {
                "type": "image_url",
                "image_url": {"url": self.url},
            }
        data_uri = f"data:{self.media_type};base64,{self.data}"
        return {
            "type": "image_url",
            "image_url": {"url": data_uri},
        }

    def to_anthropic_format(self) -> dict[str, Any]:
        """Convert to Anthropic image source block."""
        if self.source_type == "url" and self.url:
            return {
                "type": "image",
                "source": {"type": "url", "url": self.url},
            }
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": self.media_type,
                "data": self.data,
            },
        }


@dataclass
class AudioContent:
    """Audio content for multimodal messages."""

    data: str  # base64-encoded audio data
    media_type: str  # e.g. "audio/wav", "audio/mp3"

    @classmethod
    def from_file(cls, path: str | Path) -> AudioContent:
        """Create AudioContent from a file path."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        mime, _ = mimetypes.guess_type(str(p))
        if mime is None:
            mime = "audio/wav"
        raw = p.read_bytes()
        encoded = base64.b64encode(raw).decode("ascii")
        return cls(data=encoded, media_type=mime)

    @classmethod
    def from_base64(cls, data: str, media_type: str = "audio/wav") -> AudioContent:
        """Create AudioContent from a base64-encoded string."""
        return cls(data=data, media_type=media_type)


@dataclass
class MultimodalMessage:
    """Combines text and images into provider-specific message formats."""

    text: str = ""
    images: list[ImageContent] = field(default_factory=list)
    role: str = "user"

    def to_openai_messages(self) -> list[dict[str, Any]]:
        """Convert to OpenAI messages format with content array."""
        content: list[dict[str, Any]] = []
        if self.text:
            content.append({"type": "text", "text": self.text})
        for img in self.images:
            content.append(img.to_openai_format())
        return [{"role": self.role, "content": content}]

    def to_anthropic_messages(self) -> list[dict[str, Any]]:
        """Convert to Anthropic messages format with content array."""
        content: list[dict[str, Any]] = []
        if self.text:
            content.append({"type": "text", "text": self.text})
        for img in self.images:
            content.append(img.to_anthropic_format())
        return [{"role": self.role, "content": content}]
