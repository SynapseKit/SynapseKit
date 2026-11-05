"""RAG facade — 3-line happy-path entry point."""

from __future__ import annotations

import base64
import inspect
import json
import logging
import mimetypes
from collections.abc import AsyncGenerator
from contextlib import suppress
from pathlib import Path
from typing import Any

from .._audit_context import SourceAwareMessages, source_ids_from_results
from .._compat import run_sync
from ..embeddings.backend import SynapsekitEmbeddings
from ..embeddings.multimodal import BaseMultimodalEmbeddings
from ..evaluation.rag_evaluator import RAGEvaluator
from ..llm._factory import make_llm
from ..llm.multimodal import ImageContent
from ..loaders.base import Document
from ..loaders.visual import PageRenderer, VisualPage
from ..memory.conversation import ConversationMemory
from ..observability.tracer import TokenTracer
from ..retrieval.retriever import Retriever
from ..retrieval.vectorstore import InMemoryVectorStore
from ..retrieval.visual import VisualDocumentRetriever
from .pipeline import RAGConfig, RAGPipeline

logger = logging.getLogger(__name__)

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
PRESENTATION_EXTENSIONS = {".pptx", ".pptm", ".potx", ".potm", ".ppsx", ".ppsm"}


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
        evaluator: RAGEvaluator | None = None,
        context_packer: Any | None = None,
        graph_store: Any | None = None,
        visual_embeddings: BaseMultimodalEmbeddings | None = None,
        visual_retriever: VisualDocumentRetriever | None = None,
        visual_renderer: PageRenderer | None = None,
        visual_model: str | None = None,
        visual_top_k: int | None = None,
    ) -> None:
        if visual_retriever is not None and (
            visual_embeddings is not None or visual_model is not None
        ):
            raise ValueError(
                "Pass either visual_retriever or visual_embeddings/visual_model, not both"
            )
        if visual_embeddings is not None and visual_model is not None:
            raise ValueError("Pass either visual_embeddings or visual_model, not both")
        if visual_top_k is not None and visual_top_k <= 0:
            raise ValueError("visual_top_k must be positive")
        if visual_retriever is None and visual_model is not None:
            from ..embeddings.colpali import ColPaliEmbeddings

            visual_embeddings = ColPaliEmbeddings(model=visual_model)

        llm = make_llm(model, api_key, provider, system_prompt, temperature, max_tokens)
        embeddings = SynapsekitEmbeddings(model=embedding_model)
        vectorstore = InMemoryVectorStore(embeddings)
        retriever = Retriever(vectorstore, rerank=rerank)

        self._kg_builder = None
        if graph_store is not None:
            from ..retrieval.kg.builder import KnowledgeGraphBuilder
            from ..retrieval.kg.retriever import HybridKGRetriever, KGRetriever

            self._kg_builder = KnowledgeGraphBuilder(llm=llm, store=graph_store)
            kg_retriever = KGRetriever(store=graph_store, builder=self._kg_builder)
            retriever = HybridKGRetriever(vector_retriever=retriever, kg_retriever=kg_retriever)  # type: ignore

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
                evaluator=evaluator,
                context_packer=context_packer,
            )
        )
        self._embeddings = embeddings
        self._vectorstore = vectorstore
        self._visual_retriever: VisualDocumentRetriever | None
        if visual_retriever is not None:
            self._visual_retriever = visual_retriever
        elif visual_embeddings is not None:
            self._visual_retriever = VisualDocumentRetriever(
                visual_embeddings,
                renderer=visual_renderer,
            )
        else:
            self._visual_retriever = None
        self._visual_top_k = visual_top_k

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
            if self._visual_retriever is not None:
                path = Path(text)
                media_kind = self._detect_media_kind(
                    path,
                    audio_extensions=set(),
                    video_extensions=set(),
                )
                if media_kind in {"image", "pdf", "pptx"}:
                    try:
                        pages = await self._visual_retriever.render_file(path, metadata=metadata)
                    except Exception:
                        # Text/OCR ingestion remains usable when an optional
                        # renderer cannot open a file or is unavailable.
                        logger.warning(
                            "Visual rendering failed for %s; continuing with text-only ingestion",
                            path,
                            exc_info=True,
                        )
                        pages = []
                    if pages:
                        await self._visual_retriever.add_pages(pages)
            await self._pipeline.add_documents(docs)
            if self._kg_builder:
                await self._kg_builder.build_from_documents(
                    [d.text for d in docs],
                    [d.metadata.get("source", f"doc_{i}") for i, d in enumerate(docs)],
                )
            return
        await self._pipeline.add(text, metadata)
        if self._kg_builder:
            source = metadata.get("source", "doc_0") if metadata else "doc_0"
            await self._kg_builder.build_from_documents([text], [source])

    def add_documents(self, docs: list[Document]) -> None:
        """Sync: chunk and embed a list of Documents into the vectorstore."""
        run_sync(self.add_documents_async(docs))

    async def add_documents_async(self, docs: list[Document]) -> None:
        """Async: chunk and embed a list of Documents into the vectorstore."""
        visual_pages: list[VisualPage] = []
        if self._visual_retriever is not None:
            visual_pages = [
                VisualPage(
                    image=doc.metadata["image"],
                    text=doc.text,
                    metadata=doc.metadata,
                )
                for doc in docs
                if doc.metadata.get("image") is not None
            ]
            if visual_pages:
                await self._visual_retriever.add_pages(visual_pages)
        await self._pipeline.add_documents(docs)
        if self._kg_builder:
            await self._kg_builder.build_from_documents(
                [d.text for d in docs],
                [d.metadata.get("source", f"doc_{i}") for i, d in enumerate(docs)],
            )

    def add_visual_pages(self, pages: list[VisualPage]) -> None:
        """Sync: add pre-rendered visual pages and their text fallback."""
        run_sync(self.add_visual_pages_async(pages))

    async def add_visual_pages_async(self, pages: list[VisualPage]) -> None:
        """Async: add pre-rendered visual pages and their text fallback."""
        if self._visual_retriever is None:
            raise RuntimeError(
                "Configure visual_embeddings or visual_retriever before adding visual pages"
            )
        await self._visual_retriever.add_pages(pages)
        fallback_docs = [
            Document(text=page.text, metadata=dict(page.metadata))
            for page in pages
            if page.text and page.text.strip()
        ]
        if fallback_docs:
            await self._pipeline.add_documents(fallback_docs)

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
        from ..loaders.pptx import PowerPointLoader
        from ..loaders.video import SUPPORTED_EXTENSIONS as VIDEO_EXTENSIONS
        from ..loaders.video import VideoLoader

        media_kind = self._detect_media_kind(
            path,
            audio_extensions=AUDIO_EXTENSIONS,
            video_extensions=VIDEO_EXTENSIONS,
            presentation_extensions=PRESENTATION_EXTENSIONS,
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
        elif media_kind == "pptx":
            powerpoint_loader = PowerPointLoader(path=str(path))
            docs = await powerpoint_loader.aload()
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
        presentation_extensions: set[str] | None = None,
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
            if mime_type in {
                "application/vnd.ms-powerpoint",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                "application/vnd.openxmlformats-officedocument.presentationml.template",
            }:
                return "pptx"

        if suffix in IMAGE_EXTENSIONS:
            return "image"
        if suffix in PDF_EXTENSIONS:
            return "pdf"
        if suffix in (presentation_extensions or PRESENTATION_EXTENSIONS):
            return "pptx"
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
            "pptx": "slide",
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
        if self._visual_retriever is not None and self._visual_retriever.page_count:
            async for token in self._stream_visual(query, **kw):
                yield token
            return
        async for token in self._pipeline.stream(query, **kw):
            yield token

    async def ask(self, query: str, **kw) -> str:
        """Async: retrieve and answer, returns full string."""
        if self._visual_retriever is not None and self._visual_retriever.page_count:
            return await self._ask_visual(query, **kw)
        return await self._pipeline.ask(query, **kw)

    def ask_sync(self, query: str, **kw) -> str:
        """Sync: retrieve and answer (use in scripts/notebooks)."""
        return run_sync(self.ask(query, **kw))

    async def _ask_visual(self, query: str, **kw: Any) -> str:
        fallback_kw = dict(kw)
        top_k = self._visual_query_top_k(kw.pop("top_k", None))
        visual_results, text_results = await self._retrieve_visual_context(query, top_k)
        if not visual_results:
            # No visual page actually matches this query -- answer through the
            # plain text pipeline instead of framing the prompt around
            # nonexistent visual evidence.
            return await self._pipeline.ask(query, **fallback_kw)
        messages = self._build_visual_messages(query, visual_results, text_results)
        tracer = self._pipeline.config.tracer
        t0 = tracer.start_timer() if tracer else 0.0
        tokens_before = dict(self._pipeline.config.llm.tokens_used) if tracer else {}
        answer = str(await self._pipeline.config.llm.generate_with_messages(messages))
        citation_footer = self._format_visual_citations(visual_results)
        if citation_footer:
            answer = f"{answer.rstrip()}\n\n{citation_footer}"
        self._record_visual_turn(
            query,
            answer,
            contexts=self._visual_context_texts(visual_results, text_results),
            tracer=tracer,
            t0=t0,
            tokens_before=tokens_before,
        )
        return answer

    async def _stream_visual(self, query: str, **kw: Any) -> AsyncGenerator[str]:
        fallback_kw = dict(kw)
        top_k = self._visual_query_top_k(kw.pop("top_k", None))
        visual_results, text_results = await self._retrieve_visual_context(query, top_k)
        if not visual_results:
            async for token in self._pipeline.stream(query, **fallback_kw):
                yield token
            return
        messages = self._build_visual_messages(query, visual_results, text_results)
        tracer = self._pipeline.config.tracer
        t0 = tracer.start_timer() if tracer else 0.0
        tokens_before = dict(self._pipeline.config.llm.tokens_used) if tracer else {}
        answer_parts: list[str] = []
        try:
            async for token in self._pipeline.config.llm.stream_with_messages(messages):
                answer_parts.append(token)
                yield token

            citation_footer = self._format_visual_citations(visual_results)
            if citation_footer:
                suffix = f"\n\n{citation_footer}"
                answer_parts.append(suffix)
                yield suffix
        finally:
            if answer_parts:
                self._record_visual_turn(
                    query,
                    "".join(answer_parts),
                    contexts=self._visual_context_texts(visual_results, text_results),
                    tracer=tracer,
                    t0=t0,
                    tokens_before=tokens_before,
                )

    def _visual_query_top_k(self, top_k: int | None) -> int:
        value = top_k if top_k is not None else self._visual_top_k
        if value is None:
            value = self._pipeline.config.retrieval_top_k
        if value <= 0:
            raise ValueError("top_k must be positive")
        return value

    async def _retrieve_visual_context(
        self,
        query: str,
        top_k: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | list[str]]:
        if self._visual_retriever is None:
            return [], []
        visual_results = await self._visual_retriever.retrieve_with_scores(query, top_k=top_k)
        retriever = self._pipeline.config.retriever
        retrieve_with_scores = getattr(retriever, "retrieve_with_scores", None)
        if callable(retrieve_with_scores):
            result = retrieve_with_scores(query, top_k=top_k)
            text_results = await result if inspect.isawaitable(result) else result
        else:
            result = retriever.retrieve(query, top_k=top_k)
            text_results = await result if inspect.isawaitable(result) else result
        return visual_results, text_results or []

    def _build_visual_messages(
        self,
        query: str,
        visual_results: list[dict[str, Any]],
        text_results: list[dict[str, Any]] | list[str],
    ) -> list[dict[str, Any]]:
        packed_results = self._pack_visual_context(query, visual_results, text_results)
        context_blocks: list[str] = []
        for result in packed_results:
            if not result.get("_is_visual"):
                context_blocks.append(RAGPipeline._format_context_result(result))
                continue
            metadata = result.get("metadata") or {}
            source = metadata.get("source") or metadata.get("file")
            locator = metadata.get("locator")
            source_context = f"source: {source}\n" if source else ""
            locator_context = f"locator: {locator}\n" if locator else ""
            context_blocks.append(
                "[VISUAL SOURCE]\n"
                f"{source_context}"
                f"{locator_context}"
                f"page: {metadata.get('page', '?')}\n"
                f"bbox: {self._format_bbox(metadata.get('bbox', 'full-page'))}\n"
                f"score: {float(result.get('score', 0.0)):.4f}\n"
                f"text: {result.get('text', '')}\n"
                "[/VISUAL SOURCE]"
            )
        context = "\n\n".join(context_blocks) or "No context available."
        supports_images = self._llm_supports_multimodal()
        evidence_instruction = (
            "the attached page images"
            if supports_images
            else "the OCR/text context because this answer provider does not accept image messages"
        )
        prompt = (
            f"Answer the question using the OCR/text context and {evidence_instruction}. "
            "Inspect page layout, charts, tables, and visual emphasis when needed. "
            "Cite every visual claim with [page N, bbox X], using the page and bbox "
            "metadata provided below. If the evidence is insufficient, say so.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}"
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        provider = str(getattr(self._pipeline.config.llm.config, "provider", ""))
        use_anthropic = "anthropic" in provider.lower()
        if supports_images:
            for result in packed_results:
                if not result.get("_is_visual"):
                    continue
                metadata = result.get("metadata") or {}
                image = self._as_image_content(
                    result.get("image"),
                    media_type=metadata.get("image_media_type") or metadata.get("media_type"),
                )
                if image is None:
                    continue
                content.append(
                    image.to_anthropic_format() if use_anthropic else image.to_openai_format()
                )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._pipeline.config.system_prompt},
        ]
        history = self._pipeline.config.memory.format_context()
        if history:
            messages.extend(
                [
                    {"role": "user", "content": f"Previous conversation:\n{history}"},
                    {"role": "assistant", "content": "Understood."},
                ]
            )
        messages.append(
            {
                "role": "user",
                "content": content if supports_images else prompt,
            }
        )
        citation_locator_counts: dict[str, int] = {}
        source_ids_by_locator: dict[str, list[str]] = {}
        for result in packed_results:
            if not result.get("_is_visual"):
                continue
            metadata = result.get("metadata") or {}
            locator = (
                f"page {metadata.get('page', '?')}, "
                f"bbox {self._format_bbox(metadata.get('bbox', 'full-page'))}"
            )
            citation_locator_counts[locator] = citation_locator_counts.get(locator, 0) + 1
            source_ids = source_ids_from_results([result])
            if len(source_ids) == 1:
                source_ids_by_locator.setdefault(locator, []).append(source_ids[0])
        trusted_citation_ids = {
            locator: source_ids[0]
            for locator, source_ids in source_ids_by_locator.items()
            if citation_locator_counts[locator] == 1 and len(source_ids) == 1
        }
        unresolvable_citation_ids = tuple(
            locator for locator in citation_locator_counts if locator not in trusted_citation_ids
        )
        return SourceAwareMessages(
            messages,
            trusted_source_ids=source_ids_from_results(packed_results),
            trusted_citation_ids=trusted_citation_ids,
            unresolvable_citation_ids=unresolvable_citation_ids,
        )

    def _pack_visual_context(
        self,
        query: str,
        visual_results: list[dict[str, Any]],
        text_results: list[dict[str, Any]] | list[str],
    ) -> list[dict[str, Any]]:
        """Pack text and visual evidence without dropping retrieved images."""
        candidates: list[dict[str, Any]] = []
        for index, result in enumerate(text_results):
            item = dict(result) if isinstance(result, dict) else {"text": str(result)}
            raw_metadata = item.get("metadata")
            metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
            metadata.update({"_visual_context_type": "text", "_visual_context_index": index})
            item["metadata"] = metadata
            candidates.append(item)
        for index, result in enumerate(visual_results):
            item = dict(result)
            raw_metadata = item.get("metadata")
            metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
            metadata.update({"_visual_context_type": "visual", "_visual_context_index": index})
            item["metadata"] = metadata
            if not str(item.get("text", "")).strip():
                item["text"] = f"[Visual page {index + 1}]"
            candidates.append(item)

        packer = self._pipeline.config.context_packer
        packed = packer.pack(candidates, query=query) if packer is not None else candidates
        results: list[dict[str, Any]] = []
        for item in packed:
            value = dict(item) if isinstance(item, dict) else {"text": str(item)}
            raw_metadata = value.get("metadata")
            metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
            kind = metadata.pop("_visual_context_type", None)
            index = metadata.pop("_visual_context_index", None)
            value["metadata"] = metadata
            if kind == "visual" and isinstance(index, int) and index < len(visual_results):
                original = visual_results[index]
                value["text"] = original.get("text", "")
                value["image"] = original.get("image")
                value["_is_visual"] = True
            else:
                value["_is_visual"] = False
            results.append(value)
        return results

    def _llm_supports_multimodal(self) -> bool:
        return bool(getattr(self._pipeline.config.llm, "supports_multimodal", False))

    @staticmethod
    def _as_image_content(image: Any, *, media_type: str | None = None) -> ImageContent | None:
        if isinstance(image, ImageContent):
            return image
        if isinstance(image, (bytes, bytearray, memoryview)):
            encoded = base64.b64encode(bytes(image)).decode("ascii")
            detected_type = RAG._detect_image_media_type(bytes(image))
            if (
                detected_type is None
                and isinstance(media_type, str)
                and media_type.startswith("image/")
            ):
                detected_type = media_type
            return ImageContent.from_base64(encoded, media_type=detected_type or "image/png")
        if isinstance(image, (str, Path)):
            return ImageContent.from_file(image)
        return None

    @staticmethod
    def _detect_image_media_type(data: bytes) -> str | None:
        if data.startswith(bytes((137, 80, 78, 71, 13, 10, 26, 10))):
            return "image/png"
        if data.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if data.startswith((b"GIF87a", b"GIF89a")):
            return "image/gif"
        if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
            return "image/webp"
        if data.startswith(b"BM"):
            return "image/bmp"
        if data.lstrip().startswith((b"<svg", b"<?xml")):
            return "image/svg+xml"
        return None

    @staticmethod
    def _format_visual_citations(results: list[dict[str, Any]]) -> str:
        if not results:
            return ""
        lines = ["Sources:"]
        for result in results:
            metadata = result.get("metadata") or {}
            page = metadata.get("page", "?")
            bbox = RAG._format_bbox(metadata.get("bbox", "full-page"))
            source = metadata.get("source") or metadata.get("file")
            source_suffix = f", source {source}" if source else ""
            lines.append(f"- page {page}, bbox {bbox}{source_suffix}")
        return "\n".join(lines)

    @staticmethod
    def _format_bbox(bbox: Any) -> str:
        if isinstance(bbox, (list, tuple, dict)):
            return json.dumps(bbox)
        return str(bbox)

    @staticmethod
    def _visual_context_texts(
        visual_results: list[dict[str, Any]],
        text_results: list[dict[str, Any]] | list[str],
    ) -> list[str]:
        contexts = [str(result.get("text", "")) for result in visual_results]
        contexts.extend(
            RAGPipeline._format_context_result(result) if isinstance(result, dict) else str(result)
            for result in text_results
        )
        return contexts

    def _record_visual_turn(
        self,
        query: str,
        answer: str,
        *,
        contexts: list[str] | None = None,
        tracer: TokenTracer | None = None,
        t0: float = 0.0,
        tokens_before: dict[str, int] | None = None,
    ) -> None:
        self._pipeline.config.memory.add("user", query)
        self._pipeline.config.memory.add("assistant", answer)
        if tracer is None:
            return

        used = self._pipeline.config.llm.tokens_used
        before = tokens_before or {}
        input_delta = max(0, int(used["input"]) - int(before.get("input", 0)))
        output_delta = max(0, int(used["output"]) - int(before.get("output", 0)))
        call_id = tracer.record(
            input_tokens=input_delta,
            output_tokens=output_delta,
            latency_ms=tracer.elapsed_ms(t0),
        )
        if self._pipeline.config.auto_eval and tracer.enabled:
            self._pipeline._schedule_auto_eval(
                query=query,
                answer=answer,
                contexts=contexts or [],
                call_id=call_id,
            )
        if self._pipeline.config.evaluator is not None:
            self._pipeline._schedule_rag_eval(
                query=query,
                answer=answer,
                contexts=contexts or [],
                call_id=call_id,
            )

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
    def evaluator(self) -> RAGEvaluator | None:
        return self._pipeline.config.evaluator

    @property
    def memory(self) -> ConversationMemory:
        return self._pipeline.config.memory

    @property
    def visual_retriever(self) -> VisualDocumentRetriever | None:
        """Configured visual retriever, or ``None`` when text-only."""
        return self._visual_retriever

    async def wait_for_evaluations(self) -> None:
        await self._pipeline.wait_for_evaluations()

    async def wait_for_auto_eval(self) -> None:
        await self._pipeline.wait_for_auto_eval()
