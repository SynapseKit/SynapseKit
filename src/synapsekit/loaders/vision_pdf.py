"""Vision-based PDF Loader for RAG.

Extracts text and structure from complex PDFs using Vision-Language Models (VLMs).
Converts PDF pages into high-res images and passes them directly to the LLM
to output perfectly structured Markdown.
"""

from __future__ import annotations

import asyncio
import mimetypes
import os
from pathlib import Path

from synapsekit.llm.base import BaseLLM
from synapsekit.llm.multimodal import ImageContent, MultimodalMessage

from .base import Document


class VisionPDFLoader:
    """Load a PDF file and parse it into structured Markdown using a Vision-Language Model.

    This bypasses traditional text-extraction (which fails on tables/charts) and instead
    rasterizes the PDF into high-res images, passing them directly to a VLM (like GPT-4o
    or Claude 3.5 Sonnet) for perfect Markdown extraction.
    """

    def __init__(
        self,
        path: str,
        llm: BaseLLM,
        prompt: str = (
            "Extract the full text, tables, and structure from this document page. "
            "Output clean Markdown only, without any surrounding conversational text "
            "or markdown code blocks (do not wrap in ```markdown)."
        ),
        dpi: int = 150,
        max_concurrency: int = 5,
    ) -> None:
        self._path = path
        self._llm = llm
        self._prompt = prompt
        self._dpi = dpi
        self._max_concurrency = max_concurrency

    async def aload(self) -> list[Document]:
        """Asynchronously load and parse the PDF pages via VLM."""
        if not os.path.exists(self._path):
            raise FileNotFoundError(f"PDF file not found: {self._path}")

        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF required for VisionPDFLoader: pip install synapsekit[visual] or pip install pymupdf"
            ) from None

        doc = await asyncio.to_thread(fitz.open, self._path)
        media_type, _ = mimetypes.guess_type(self._path)
        source_name = Path(self._path).name
        total_pages = len(doc)

        docs: list[Document | None] = [None] * total_pages
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def _process_page(page_index: int) -> None:
            async with semaphore:
                page = doc.load_page(page_index)
                pix = await asyncio.to_thread(page.get_pixmap, dpi=self._dpi)
                image_bytes = pix.tobytes("png")

                # Convert to base64, then to ImageContent
                import base64

                b64_image = base64.b64encode(image_bytes).decode("ascii")
                img_content = ImageContent.from_base64(b64_image, media_type="image/png")

                msg = MultimodalMessage(text=self._prompt, images=[img_content], role="user")

                provider = getattr(self._llm.config, "provider", "")
                if provider == "anthropic":
                    messages = msg.to_anthropic_messages()
                elif provider in ("gemini", "google"):
                    # Gemini usually supports OpenAI vision format in litellm/SDKs, or natively via parts
                    # We will fallback to OpenAI format which is the standard cross-provider multimodal payload
                    messages = msg.to_openai_messages()
                else:
                    messages = msg.to_openai_messages()

                text = await self._llm.generate_with_messages(messages)
                text = text.strip()
                if text.startswith("```markdown"):
                    text = text[11:]
                if text.startswith("```"):
                    text = text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

                page_number = page_index + 1
                docs[page_index] = Document(
                    text=text,
                    metadata={
                        "source": self._path,
                        "file": self._path,
                        "source_type": "pdf",
                        "media_type": media_type or "application/pdf",
                        "loader": "VisionPDFLoader",
                        "chunk_type": "page",
                        "page": page_number,
                        "locator": f"{source_name} page {page_number}",
                    },
                )

        tasks = [_process_page(i) for i in range(total_pages)]
        await asyncio.gather(*tasks)

        doc.close()

        # Filter out None in case any failed (though exceptions will raise out of gather)
        return [d for d in docs if d is not None]

    def load(self) -> list[Document]:
        """Synchronously load and parse the PDF pages via VLM."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError(
                "Cannot call load() synchronously from within a running event loop. Use aload() instead."
            )

        return asyncio.run(self.aload())
