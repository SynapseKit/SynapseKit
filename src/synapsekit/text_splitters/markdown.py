from __future__ import annotations

import re
from typing import Any

from .base import BaseSplitter
from .recursive import RecursiveCharacterTextSplitter


class MarkdownTextSplitter(BaseSplitter):
    """Split markdown text respecting document structure.

    Splits on markdown headers while preserving parent header context
    in each resulting chunk. Falls back to recursive character splitting
    for oversized sections.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        headers_to_split_on: list[tuple[str, str]] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.headers_to_split_on = headers_to_split_on or [
            ("####", "Header4"),
            ("###", "Header3"),
            ("##", "Header2"),
            ("#", "Header1"),
        ]
        # Sort by length descending so #### matches before #
        self.headers_to_split_on.sort(key=lambda x: len(x[0]), reverse=True)

    def split(self, text: str) -> list[str]:
        """Split *text* into a list of chunks."""
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        sections = self._split_by_headers(text)
        chunks: list[str] = []

        fallback = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["---", "\n\n", "\n", ". ", " "],
        )

        for section in sections:
            headers: dict[str, str] = section["headers"]
            content: str = section["content"]
            prefix = self._format_header_context(headers)
            body = content.strip()
            if not body:
                continue

            full = (prefix + body) if prefix else body

            if len(full) <= self.chunk_size:
                chunks.append(full)
            else:
                # Fall back to recursive splitting on the body, then
                # prepend the header context to each sub-chunk.
                sub_chunks = fallback.split(body)
                for sc in sub_chunks:
                    candidate = (prefix + sc) if prefix else sc
                    chunks.append(candidate)

        # Apply overlap between chunks
        if self.chunk_overlap > 0 and len(chunks) >= 2:
            overlapped = [chunks[0]]
            for i in range(1, len(chunks)):
                tail = chunks[i - 1][-self.chunk_overlap :]
                overlapped.append(tail + chunks[i])
            chunks = overlapped

        return chunks

    def _split_by_headers(self, text: str) -> list[dict[str, Any]]:
        """Split text into sections, each with its header metadata.

        Returns a list of dicts:
        ``{"content": str, "headers": dict[str, str]}``
        """
        # Build a regex that matches any of the configured header markers
        # at the start of a line.  E.g. ^(####|###|##|#)\s+(.+)$
        markers = [re.escape(h[0]) for h in self.headers_to_split_on]
        header_re = re.compile(
            r"^(" + "|".join(markers) + r")\s+(.+)$", re.MULTILINE
        )

        # Map marker string -> label name  (e.g. "##" -> "Header2")
        marker_to_label: dict[str, str] = {
            h[0]: h[1] for h in self.headers_to_split_on
        }

        sections: list[dict[str, Any]] = []
        current_headers: dict[str, str] = {}
        last_end = 0

        for match in header_re.finditer(text):
            # Flush content before this header
            preceding = text[last_end : match.start()]
            if preceding.strip():
                sections.append(
                    {
                        "content": preceding,
                        "headers": dict(current_headers),
                    }
                )

            marker = match.group(1)
            header_text = match.group(2).strip()
            label = marker_to_label[marker]

            # Update header stack: set this level and clear deeper levels
            current_headers[label] = f"{marker} {header_text}"
            # Clear all labels that are "deeper" (longer marker)
            for m, lbl in self.headers_to_split_on:
                if len(m) > len(marker):
                    current_headers.pop(lbl, None)

            last_end = match.end()

        # Flush remaining content
        remaining = text[last_end:]
        if remaining.strip():
            sections.append(
                {
                    "content": remaining,
                    "headers": dict(current_headers),
                }
            )

        # If no headers were found at all, return the whole text as one section
        if not sections:
            sections.append({"content": text, "headers": {}})

        return sections

    @staticmethod
    def _format_header_context(headers: dict[str, str]) -> str:
        """Format header hierarchy as a prefix string.

        Produces output like ``"# Title\\n## Section\\n"`` with headers
        ordered from shallowest to deepest.
        """
        if not headers:
            return ""
        # Sort by label so Header1 < Header2 < Header3 < Header4
        ordered = sorted(headers.items(), key=lambda x: x[0])
        return "\n".join(v for _, v in ordered) + "\n"
