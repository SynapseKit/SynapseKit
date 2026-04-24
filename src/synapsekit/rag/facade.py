"""RAG facade — 3-line happy-path entry point."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path

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
        from ..loaders.video import SUPPORTED_EXTENSIONS as VIDEO_EXTENSIONS
        from ..loaders.video import VideoLoader

        suffix = path.suffix.lower()
        llm = self._pipeline.config.llm

        if suffix in IMAGE_EXTENSIONS:
            prompt = kwargs.get("caption") or kwargs.get("prompt")
            image_loader = ImageLoader(
                path=path,
                llm=llm,
                prompt=prompt or "Describe this image in detail for retrieval.",
            )
            docs = await image_loader.aload()
        elif suffix in AUDIO_EXTENSIONS:
            audio_loader = AudioLoader(
                path=str(path),
                api_key=kwargs.get("audio_api_key", llm.config.api_key),
                backend=kwargs.get("audio_backend", "whisper_api"),
                language=kwargs.get("language"),
                model=kwargs.get("audio_model", "whisper-1"),
            )
            docs = await audio_loader.aload()
        elif suffix in VIDEO_EXTENSIONS:
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
        else:
            return None

        if metadata:
            for doc in docs:
                doc.metadata = {**doc.metadata, **metadata}
        return docs

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
