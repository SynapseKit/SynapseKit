"""RAG facade — 3-line happy-path entry point."""

from __future__ import annotations

import mimetypes
from collections.abc import AsyncGenerator
from contextlib import suppress
from pathlib import Path
from typing import Any

from .._compat import run_sync
from ..embeddings.backend import SynapsekitEmbeddings
from ..llm._factory import make_llm
from ..loaders.base import Document
from ..memory.conversation import ConversationMemory
from ..observability.tracer import TokenTracer
from ..retrieval.retriever import Retriever
from ..retrieval.vectorstore import InMemoryVectorStore
from .pipeline import RAGConfig, RAGPipeline

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
    ".heic",
    ".heif",
    ".svg",
}
PDF_EXTENSIONS = {".pdf"}


class RAG:
    """
    3-line RAG facade with sane defaults.

    Example::

        rag = RAG(model="gpt-4o-mini", api_key="sk-...")
        rag.add("Your document text here")
        answer = rag.ask_sync("What is the main topic?")
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        provider: str | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        rerank: bool = False,
        memory_window: int = 10,
        retrieval_top_k: int = 5,
        system_prompt: str = "Answer using only the provided context. If the context does not contain the answer, say so.",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        trace: bool = True,
        auto_eval: bool = False,
        context_packer: Any | None = None,
    ) -> None:
        llm = make_llm(model, api_key, provider, system_prompt, temperature, max_tokens)
        embeddings = SynapsekitEmbeddings(model=embedding_model)
        vectorstore = InMemoryVectorStore(embeddings)
        retriever = Retriever(vectorstore, rerank=rerank)
        memory = ConversationMemory(window=memory_window)
        tracer = TokenTracer(model=model, enabled=trace)

        self._pipeline = RAGPipeline(
            RAGConfig(
                llm=llm,
                retriever=retriever,
                memory=memory,
                tracer=tracer,
                retrieval_top_k=retrieval_top_k,
                system_prompt=system_prompt,
                auto_eval=auto_eval,
                context_packer=context_packer,
            )
        )
        self._embeddings = embeddings
        self._vectorstore = vectorstore

    # ------------------------------------------------------------------ #
    # Document ingestion
    # ------------------------------------------------------------------ #

    def add(self, text: str, metadata: dict | None = None, **kwargs) -> None:
        """Sync: add raw text, or auto-detect multimodal file paths."""
        run_sync(self.add_async(text, metadata=metadata, **kwargs))

    async def add_async(self, text: str, metadata: dict | None = None, **kwargs) -> None:
        """Async: add raw text, or auto-detect multimodal file paths."""
        docs = await self._load_multimodal_documents(text, metadata=metadata, **kwargs)
        if docs is not None:
            await self._pipeline.add_documents(docs)
            return
        await self._pipeline.add(text, metadata)

    def add_documents(self, docs: list[Document]) -> None:
        """Sync: chunk and embed a list of Documents into the vectorstore."""
        run_sync(self._pipeline.add_documents(docs))

    async def add_documents_async(self, docs: list[Document]) -> None:
        """Async: chunk and embed a list of Documents into the vectorstore."""
        await self._pipeline.add_documents(docs)

    async def _load_multimodal_documents(
        self,
        text: str,
        metadata: dict | None = None,
        **kwargs,
    ) -> list[Document] | None:
        path = Path(text)
        if not path.exists() or not path.is_file():
            return None

        from ..loaders.audio import SUPPORTED_EXTENSIONS as AUDIO_EXTENSIONS
        from ..loaders.audio import AudioLoader
        from ..loaders.image import ImageLoader
        from ..loaders.pdf import PDFLoader
        from ..loaders.video import SUPPORTED_EXTENSIONS as VIDEO_EXTENSIONS
        from ..loaders.video import VideoLoader

        media_kind = self._detect_media_kind(
            path,
            audio_extensions=AUDIO_EXTENSIONS,
            video_extensions=VIDEO_EXTENSIONS,
        )
        llm = self._pipeline.config.llm

        if media_kind == "image":
            prompt = kwargs.get("caption") or kwargs.get("prompt")
            image_loader = ImageLoader(
                path=path,
                llm=llm,
                prompt=prompt or "Describe this image in detail for retrieval.",
            )
            docs = await image_loader.aload()
        elif media_kind == "audio":
            audio_loader = AudioLoader(
                path=str(path),
                api_key=kwargs.get("audio_api_key", llm.config.api_key),
                backend=kwargs.get("audio_backend", "whisper_api"),
                language=kwargs.get("language"),
                model=kwargs.get("audio_model", "whisper-1"),
            )
            docs = await audio_loader.aload()
        elif media_kind == "video":
            video_loader = VideoLoader(
                path=str(path),
                api_key=kwargs.get("audio_api_key", llm.config.api_key),
                backend=kwargs.get("audio_backend", "whisper_api"),
                language=kwargs.get("language"),
                keep_audio=bool(kwargs.get("keep_audio", False)),
                llm=llm,
                frame_interval=kwargs.get("frame_interval", 30),
                frame_prompt=kwargs.get(
                    "frame_prompt",
                    "Describe this video frame in detail for retrieval.",
                ),
                keep_frames=bool(kwargs.get("keep_frames", False)),
            )
            docs = await video_loader.aload()
        elif media_kind == "pdf":
            pdf_loader = PDFLoader(path=str(path))
            docs = await pdf_loader.aload()
        else:
            return None

        for doc in docs:
            merged_metadata = {**doc.metadata, **(metadata or {})}
            doc.metadata = self._normalize_document_metadata(
                path=path,
                source_type=media_kind,
                metadata=merged_metadata,
            )
        return docs

    @staticmethod
    def _detect_media_kind(
        path: Path,
        *,
        audio_extensions: set[str],
        video_extensions: set[str],
    ) -> str | None:
        suffix = path.suffix.lower()
        mime_type, _ = mimetypes.guess_type(str(path))

        if mime_type:
            if mime_type.startswith("image/"):
                return "image"
            if mime_type.startswith("audio/"):
                return "audio"
            if mime_type.startswith("video/"):
                return "video"
            if mime_type == "application/pdf":
                return "pdf"

        if suffix in IMAGE_EXTENSIONS:
            return "image"
        if suffix in PDF_EXTENSIONS:
            return "pdf"
        if suffix in audio_extensions:
            return "audio"
        if suffix in video_extensions:
            return "video"
        return None

    @staticmethod
    def _normalize_document_metadata(
        path: Path, source_type: str, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        normalized = dict(metadata)
        mime_type, _ = mimetypes.guess_type(str(path))

        normalized.setdefault("source", str(path))
        normalized.setdefault("file", str(path))
        normalized.setdefault("source_type", source_type)
        if mime_type:
            normalized.setdefault("media_type", mime_type)
        normalized.setdefault("chunk_type", RAG._default_chunk_type(source_type))

        if normalized.get("page") is not None:
            with suppress(TypeError, ValueError):
                normalized["page"] = int(normalized["page"])

        if normalized.get("locator") is None:
            normalized["locator"] = RAG._build_locator(path, normalized)
        return normalized

    @staticmethod
    def _default_chunk_type(source_type: str) -> str:
        return {
            "audio": "transcript",
            "image": "image_caption",
            "pdf": "page",
            "video": "transcript",
        }.get(source_type, "text")

    @staticmethod
    def _build_locator(path: Path, metadata: dict[str, Any]) -> str | None:
        page = metadata.get("page")
        if page is not None:
            return f"{path.name} page {page}"

        start_time = RAG._to_float(metadata.get("start_time"))
        end_time = RAG._to_float(metadata.get("end_time"))
        timestamp = RAG._to_float(metadata.get("timestamp"))
        if start_time is not None:
            if end_time is not None and end_time != start_time:
                return f"{RAG._format_seconds(start_time)}-{RAG._format_seconds(end_time)}"
            return RAG._format_seconds(start_time)
        if timestamp is not None:
            return RAG._format_seconds(timestamp)

        frame_index = metadata.get("frame_index")
        if frame_index is not None:
            return f"{path.name} frame {frame_index}"

        return path.name

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _format_seconds(value: float) -> str:
        total_seconds = max(0, int(value))
        hours, rem = divmod(total_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    # ------------------------------------------------------------------ #
    # Querying
    # ------------------------------------------------------------------ #

    async def stream(self, query: str, **kw) -> AsyncGenerator[str]:
        """Async generator that yields tokens as they arrive from the LLM."""
        async for token in self._pipeline.stream(query, **kw):
            yield token

    async def ask(self, query: str, **kw) -> str:
        """Async: retrieve and answer, returns full string."""
        return await self._pipeline.ask(query, **kw)

    def ask_sync(self, query: str, **kw) -> str:
        """Sync: retrieve and answer (use in scripts/notebooks)."""
        return run_sync(self._pipeline.ask(query, **kw))

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Persist the vectorstore to a .npz file."""
        self._vectorstore.save(path)

    def load(self, path: str) -> None:
        """Load a previously saved vectorstore from a .npz file."""
        self._vectorstore.load(path)

    # ------------------------------------------------------------------ #
    # Observability
    # ------------------------------------------------------------------ #

    @property
    def tracer(self) -> TokenTracer | None:
        return self._pipeline.config.tracer

    @property
    def memory(self) -> ConversationMemory:
        return self._pipeline.config.memory
