"""Tests for the RAG facade — 3-line API."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synapsekit import RAG
from synapsekit.loaders.base import Document
from synapsekit.memory.conversation import ConversationMemory
from synapsekit.observability.tracer import TokenTracer


def _patch_rag(rag: RAG, tokens=("Answer", " here")):
    """Replace the pipeline's LLM and retriever with mocks."""
    retriever = rag._pipeline.config.retriever
    retriever.retrieve = AsyncMock(return_value=["context"])
    retriever.retrieve_with_scores = AsyncMock(
        return_value=[
            {
                "text": "context",
                "score": 0.9,
                "metadata": {"source_type": "text", "locator": "context"},
            }
        ]
    )

    # HybridKGRetriever wraps the standard retriever in _vector_retriever
    if hasattr(retriever, "_vector_retriever"):
        retriever._vector_retriever._store.add = AsyncMock()
    elif hasattr(retriever, "_store"):
        retriever._store.add = AsyncMock()

    async def mock_stream(messages, **kw):
        for t in tokens:
            yield t

    rag._pipeline.config.llm.stream_with_messages = mock_stream
    rag._pipeline.config.llm._input_tokens = 5
    rag._pipeline.config.llm._output_tokens = 3

    mock_splitter = MagicMock()
    mock_splitter.split = MagicMock(return_value=["chunk"])
    rag._pipeline._splitter = mock_splitter


class TestRAGFacade:
    def test_init_openai_auto_detect(self):
        with patch("synapsekit.llm.openai.OpenAILLM.__init__", return_value=None):
            rag = RAG(model="gpt-4o-mini", api_key="sk-test")
            assert rag._pipeline.config.llm is not None

    def test_init_anthropic_auto_detect(self):
        with patch("synapsekit.llm.anthropic.AnthropicLLM.__init__", return_value=None):
            rag = RAG(model="claude-haiku-4-5-20251001", api_key="sk-test")
            assert rag._pipeline.config.llm is not None

    def test_init_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            RAG(model="gpt-4o-mini", api_key="sk-test", provider="unknownxyz")

    def test_add_sync(self):
        rag = RAG(model="gpt-4o-mini", api_key="sk-test")
        _patch_rag(rag)
        rag.add("Some document text.")
        rag._pipeline._splitter.split.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_async(self):
        rag = RAG(model="gpt-4o-mini", api_key="sk-test")
        _patch_rag(rag)
        await rag.add_async("Some document text.")
        rag._pipeline._splitter.split.assert_called_once()

    def test_add_sync_autodetects_image_file(self, tmp_path):
        image_file = tmp_path / "diagram.png"
        image_file.write_bytes(b"png")

        rag = RAG(model="gpt-4o-mini", api_key="sk-test")
        _patch_rag(rag)

        with patch(
            "synapsekit.loaders.image.ImageLoader.aload",
            new=AsyncMock(
                return_value=[Document(text="caption", metadata={"source_type": "image"})]
            ),
        ):
            rag.add(str(image_file))

        rag._pipeline._splitter.split.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_async_autodetects_audio_file(self, tmp_path):
        audio_file = tmp_path / "meeting.mp3"
        audio_file.write_bytes(b"audio")

        rag = RAG(model="gpt-4o-mini", api_key="sk-test")
        _patch_rag(rag)

        with patch(
            "synapsekit.loaders.audio.AudioLoader.aload",
            new=AsyncMock(
                return_value=[Document(text="transcript", metadata={"source_type": "audio"})]
            ),
        ):
            await rag.add_async(str(audio_file))

        rag._pipeline._splitter.split.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_async_autodetects_pdf_file(self, tmp_path):
        pdf_file = tmp_path / "report.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        rag = RAG(model="gpt-4o-mini", api_key="sk-test")
        _patch_rag(rag)

        with patch(
            "synapsekit.loaders.pdf.PDFLoader.aload",
            new=AsyncMock(
                return_value=[
                    Document(text="page text", metadata={"page": 2, "source_type": "pdf"})
                ]
            ),
        ):
            await rag.add_async(str(pdf_file))

        rag._pipeline._splitter.split.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_async_uses_mime_type_for_webm_video(self, tmp_path):
        video_file = tmp_path / "demo.webm"
        video_file.write_bytes(b"video")

        rag = RAG(model="gpt-4o-mini", api_key="sk-test")
        _patch_rag(rag)

        with (
            patch("synapsekit.rag.facade.mimetypes.guess_type", return_value=("video/webm", None)),
            patch(
                "synapsekit.loaders.video.VideoLoader.aload",
                new=AsyncMock(
                    return_value=[Document(text="frame text", metadata={"timestamp": 12.0})]
                ),
            ) as video_aload,
            patch("synapsekit.loaders.audio.AudioLoader.aload", new=AsyncMock()) as audio_aload,
        ):
            await rag.add_async(str(video_file))

        assert video_aload.await_count == 1
        assert audio_aload.await_count == 0

    @pytest.mark.asyncio
    async def test_add_async_existing_non_multimodal_path_falls_back_to_text(self, tmp_path):
        text_file = tmp_path / "notes.txt"
        text_file.write_text("hello", encoding="utf-8")

        rag = RAG(model="gpt-4o-mini", api_key="sk-test")
        rag._pipeline.add = AsyncMock()

        await rag.add_async(str(text_file))

        rag._pipeline.add.assert_awaited_once_with(str(text_file), None)

    @pytest.mark.asyncio
    async def test_add_async_normalizes_routed_document_metadata(self, tmp_path):
        pdf_file = tmp_path / "report.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        rag = RAG(model="gpt-4o-mini", api_key="sk-test")
        rag._pipeline.add_documents = AsyncMock()

        with patch(
            "synapsekit.loaders.pdf.PDFLoader.aload",
            new=AsyncMock(return_value=[Document(text="page text", metadata={"page": 3})]),
        ):
            await rag.add_async(str(pdf_file), metadata={"topic": "roadmap"})

        docs = rag._pipeline.add_documents.await_args.args[0]
        assert len(docs) == 1
        assert docs[0].metadata["source_type"] == "pdf"
        assert docs[0].metadata["chunk_type"] == "page"
        assert docs[0].metadata["locator"] == "report.pdf page 3"
        assert docs[0].metadata["topic"] == "roadmap"

    def test_add_image_caption_kw_passed_to_loader(self, tmp_path):
        image_file = tmp_path / "diagram.png"
        image_file.write_bytes(b"png")

        rag = RAG(model="gpt-4o-mini", api_key="sk-test")
        _patch_rag(rag)

        with patch("synapsekit.loaders.image.ImageLoader") as mock_loader_cls:
            mock_loader = mock_loader_cls.return_value
            mock_loader.aload = AsyncMock(return_value=[Document(text="caption", metadata={})])

            rag.add(str(image_file), caption="Login flow from Q4 review")

            assert mock_loader_cls.call_args.kwargs["prompt"] == "Login flow from Q4 review"

    @pytest.mark.asyncio
    async def test_ask_returns_string(self):
        rag = RAG(model="gpt-4o-mini", api_key="sk-test")
        _patch_rag(rag)
        answer = await rag.ask("What is the topic?")
        assert answer == "Answer here"

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        rag = RAG(model="gpt-4o-mini", api_key="sk-test")
        _patch_rag(rag, tokens=["A", "B", "C"])
        tokens = []
        async for t in rag.stream("question?"):
            tokens.append(t)
        assert tokens == ["A", "B", "C"]

    def test_ask_sync(self):
        rag = RAG(model="gpt-4o-mini", api_key="sk-test")
        _patch_rag(rag)
        answer = rag.ask_sync("question?")
        assert answer == "Answer here"

    def test_tracer_property(self):
        rag = RAG(model="gpt-4o-mini", api_key="sk-test")
        assert isinstance(rag.tracer, TokenTracer)

    def test_memory_property(self):
        rag = RAG(model="gpt-4o-mini", api_key="sk-test")
        assert isinstance(rag.memory, ConversationMemory)

    def test_save_raises_when_empty(self, tmp_path):
        rag = RAG(model="gpt-4o-mini", api_key="sk-test")
        with pytest.raises(ValueError, match="empty"):
            rag.save(str(tmp_path / "store.npz"))

    @pytest.mark.asyncio
    async def test_trace_disabled(self):
        rag = RAG(model="gpt-4o-mini", api_key="sk-test", trace=False)
        _patch_rag(rag)
        await rag.ask("test")
        assert rag.tracer.summary()["calls"] == 0

    def test_auto_eval_flag_propagates_to_pipeline(self):
        rag = RAG(model="gpt-4o-mini", api_key="sk-test", auto_eval=True)
        assert rag._pipeline.config.auto_eval is True

    @pytest.mark.asyncio
    async def test_graph_store_triggers_builder_on_add(self):
        pytest.importorskip("networkx")
        from synapsekit.retrieval.kg.backends import NetworkXStore

        store = NetworkXStore()
        rag = RAG(model="gpt-4o-mini", api_key="sk-test", graph_store=store)
        rag._pipeline.add = AsyncMock()
        assert rag._kg_builder is not None
        rag._kg_builder.build_from_documents = AsyncMock()

        await rag.add_async("Graph doc", metadata={"source": "doc-99"})

        rag._pipeline.add.assert_awaited_once_with("Graph doc", {"source": "doc-99"})
        rag._kg_builder.build_from_documents.assert_awaited_once_with(["Graph doc"], ["doc-99"])

    @pytest.mark.asyncio
    async def test_graph_retrieval_integration_on_ask(self):
        pytest.importorskip("networkx")
        from synapsekit.retrieval.kg.backends import NetworkXStore
        from synapsekit.retrieval.kg.retriever import HybridKGRetriever

        store = NetworkXStore()
        rag = RAG(model="gpt-4o-mini", api_key="sk-test", graph_store=store)
        _patch_rag(rag)
        rag._pipeline.run = AsyncMock(return_value={"answer": "Graph Answer"})

        await rag.ask("test query")

        assert isinstance(rag._pipeline.config.retriever, HybridKGRetriever)
