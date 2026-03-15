from __future__ import annotations

from typing import Any

from ..base import BaseTool, ToolResult


class PDFReaderTool(BaseTool):
    """Read and extract text from PDF files."""

    name = "pdf_reader"
    description = (
        "Read and extract text from a PDF file. "
        "Optionally specify page numbers to read specific pages."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the PDF file",
            },
            "page_numbers": {
                "type": "string",
                "description": "Comma-separated page numbers to extract (e.g. '1,3,5'). "
                "If not provided, all pages are extracted.",
                "default": "",
            },
        },
        "required": ["file_path"],
    }

    async def run(
        self,
        file_path: str = "",
        page_numbers: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        file_path = file_path or kwargs.get("input", "")
        if not file_path:
            return ToolResult(output="", error="No file path provided.")

        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError(
                "pypdf required for PDFReaderTool: pip install synapsekit[pdf]"
            ) from None

        import os

        if not os.path.isfile(file_path):
            return ToolResult(output="", error=f"File not found: {file_path}")

        try:
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)

            if page_numbers:
                try:
                    indices = [int(p.strip()) - 1 for p in page_numbers.split(",")]
                except ValueError:
                    return ToolResult(output="", error=f"Invalid page numbers: {page_numbers}")
                # Validate range
                for idx in indices:
                    if idx < 0 or idx >= total_pages:
                        return ToolResult(
                            output="",
                            error=f"Page {idx + 1} out of range (1-{total_pages}).",
                        )
            else:
                indices = list(range(total_pages))

            pages = []
            for idx in indices:
                text = reader.pages[idx].extract_text() or ""
                pages.append(f"--- Page {idx + 1} ---\n{text}")

            return ToolResult(output="\n\n".join(pages))
        except Exception as e:
            return ToolResult(output="", error=f"PDF read failed: {e}")
