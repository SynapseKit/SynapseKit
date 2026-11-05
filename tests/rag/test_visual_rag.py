from __future__ import annotations

import numpy as np
import pytest

from synapsekit import RAG
from synapsekit import MultimodalRAG as TopLevelMultimodalRAG
from synapsekit.audit import AuditTracer, EventKind, VerifiableAgent, build_provenance
from synapsekit.audit.provenance import source_id_hash
from synapsekit.embeddings.multimodal import BaseMultimodalEmbeddings
from synapsekit.loaders.base import Document
from synapsekit.loaders.pdf import PDFLoader
from synapsekit.loaders.visual import VisualPage
from synapsekit.rag.multimodal import MultimodalRAG
from synapsekit.retrieval.visual import VisualDocumentRetriever


class FakeVisualEmbeddings(BaseMultimodalEmbeddings):
    dimensions = 2

    @staticmethod
    def _vector(value: object) -> np.ndarray:
        text = str(value).lower()
        if "answer" in text or "page-a" in text:
            return np.array([[1.0, 0.0]], dtype=np.float32)
        return np.array([[0.0, 1.0]], dtype=np.float32)

    async def _embed_images(self, images):
        return [self._vector(image) for image in images]

    async def _embed_texts(self, texts):
        return [self._vector(text) for text in texts]


class FakeRenderer:
    async def render(self, path):
        return [
            VisualPage(
                image=b"page-a",
                text="The answer is in the highlighted cell.",
                metadata={"page": 4, "bbox": [12.0, 24.0, 120.0, 240.0]},
            )
        ]


class BrokenRenderer:
    async def render(self, path):
        raise RuntimeError("renderer failed")


class RecordingPacker:
    def __init__(self):
        self.calls = []

    def pack(self, chunks, query=None):
        self.calls.append((chunks, query))
        return [chunks[-1]]


class RecordingAddDocuments:
    """Stand-in for RAGPipeline.add_documents that records its calls."""

    def __init__(self):
        self.await_args = None
        self.await_count = 0

    async def __call__(self, *args, **kwargs):
        self.await_args = _Args(args, kwargs)
        self.await_count += 1


class _Args:
    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs


class NullRetriever:
    """Stand-in retriever whose scored-retrieval path always returns nothing."""

    async def retrieve_with_scores(self, *args, **kwargs):
        return []


def _make_rag(**kwargs):
    rag = RAG(model="gpt-4o-mini", api_key="[REDACTED]", **kwargs)
    rag._pipeline.add_documents = RecordingAddDocuments()
    rag._pipeline.config.retriever = NullRetriever()
    return rag


@pytest.mark.asyncio
async def test_rag_visual_backend_indexes_visual_pages_and_text_fallback(tmp_path, monkeypatch):
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"pdf")
    rag = _make_rag(
        visual_embeddings=FakeVisualEmbeddings(),
        visual_renderer=FakeRenderer(),
    )

    async def fake_aload(self):
        return [Document(text="OCR fallback", metadata={"page": 4})]

    monkeypatch.setattr(PDFLoader, "aload", fake_aload)
    await rag.add_async(str(pdf_path))

    assert rag.visual_retriever is not None
    assert rag.visual_retriever.page_count == 1
    fallback_docs = rag._pipeline.add_documents.await_args.args[0]
    assert fallback_docs[0].text == "OCR fallback"


@pytest.mark.asyncio
async def test_rag_visual_ask_sends_page_images_and_appends_citations():
    rag = _make_rag(visual_embeddings=FakeVisualEmbeddings())
    await rag.visual_retriever.add_pages(
        [
            VisualPage(
                image=b"page-a",
                text="The answer is in the highlighted cell.",
                metadata={
                    "source_id": "visual-doc-4",
                    "source": "report.pdf",
                    "page": 4,
                    "bbox": [12.0, 24.0, 120.0, 240.0],
                },
            )
        ]
    )

    captured = {}

    async def generate_with_messages(messages, **kwargs):
        captured["messages"] = messages
        return "The answer is 42."

    rag._pipeline.config.llm.generate_with_messages = generate_with_messages
    answer = await rag.ask("What is the answer?")

    content = captured["messages"][-1]["content"]
    assert captured["messages"]._trusted_source_ids == ("visual-doc-4",)
    assert all("trusted_source_ids" not in message for message in captured["messages"])
    assert isinstance(content, list)
    assert any(block.get("type") == "image_url" for block in content)
    prompt_text = next(block["text"] for block in content if block.get("type") == "text")
    assert "page: 4" in prompt_text
    assert "bbox: [12.0, 24.0, 120.0, 240.0]" in prompt_text
    assert "source: report.pdf" in prompt_text
    assert "page 4" in answer
    assert "bbox [12.0, 24.0, 120.0, 240.0]" in answer
    assert "source report.pdf" in answer


async def _ask_audited_visual_rag(pages, answer):
    rag = _make_rag(visual_embeddings=FakeVisualEmbeddings())
    await rag.visual_retriever.add_pages(pages)
    captured = {}

    async def generate_with_messages(messages, **kwargs):
        captured["messages"] = messages
        return answer

    rag._pipeline.config.llm.generate_with_messages = generate_with_messages
    tracer = AuditTracer(run_id="audited-visual-rag")
    agent = VerifiableAgent(
        rag,
        tracer=tracer,
        instrument_attrs=["_visual_retriever", "_pipeline.config.llm"],
    )
    actual_answer = await agent.ask("What does the page show?")
    return actual_answer, tracer, captured["messages"]


@pytest.mark.asyncio
async def test_runtime_visual_citation_resolves_to_its_retrieved_source():
    locator = "page 4, bbox [12.0, 24.0, 120.0, 240.0]"
    answer, tracer, provider_messages = await _ask_audited_visual_rag(
        [
            VisualPage(
                image=b"page-a",
                text="The answer is in the highlighted cell.",
                metadata={
                    "source_id": "visual-doc-4",
                    "source": "report.pdf",
                    "page": 4,
                    "bbox": [12.0, 24.0, 120.0, 240.0],
                },
            )
        ],
        f"The cited value is 42 [{locator}].",
    )

    provenance = build_provenance(tracer.records)
    source_entity_id = next(
        identifier
        for identifier, entity in provenance["entity"].items()
        if entity.get("sk:sourceIdHash") == source_id_hash("visual-doc-4")
    )
    response = next(
        record for record in tracer.records if record.kind == EventKind.LLM_RESPONSE.value
    )
    response_entity_id = next(
        identifier
        for identifier, entity in provenance["entity"].items()
        if entity.get("sk:eventIdHash") == source_id_hash(response.event_id)
    )
    response_activity = next(
        activity
        for activity in provenance["activity"].values()
        if activity.get("sk:eventIdHash") == source_id_hash(response.event_id)
    )
    llm_call = next(record for record in tracer.records if record.kind == EventKind.LLM_CALL.value)
    prompt_text = next(
        block["text"]
        for message in provider_messages
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if block.get("type") == "text"
    )

    assert answer.startswith("The cited value is 42")
    assert llm_call.payload.get("trusted_citation_ids") == {locator: "visual-doc-4"}
    assert response_activity.get("sk:lineageStatus") == "resolved"
    assert any(
        relation["prov:generatedEntity"] == response_entity_id
        and relation["prov:usedEntity"] == source_entity_id
        for relation in provenance["wasDerivedFrom"].values()
    )
    assert "trusted_citation_ids" not in response.payload
    assert "visual-doc-4" not in prompt_text
    assert all(
        "trusted_source_ids" not in message and "trusted_citation_ids" not in message
        for message in provider_messages
    )


@pytest.mark.asyncio
async def test_runtime_visual_citation_with_duplicate_locators_stays_unresolved():
    locator = "page 4, bbox [12.0, 24.0, 120.0, 240.0]"
    _answer, tracer, provider_messages = await _ask_audited_visual_rag(
        [
            VisualPage(
                image=b"page-a",
                text="First candidate.",
                metadata={
                    "source_id": locator,
                    "page": 4,
                    "bbox": [12.0, 24.0, 120.0, 240.0],
                },
            ),
            VisualPage(
                image=b"page-b",
                text="Second candidate.",
                metadata={
                    "source_id": "visual-doc-b",
                    "page": 4,
                    "bbox": [12.0, 24.0, 120.0, 240.0],
                },
            ),
        ],
        f"The cited value is ambiguous [{locator}].",
    )

    provenance = build_provenance(tracer.records)
    source_entity_ids = {
        identifier
        for identifier, entity in provenance["entity"].items()
        if entity.get("sk:sourceIdHash")
        in {source_id_hash(locator), source_id_hash("visual-doc-b")}
    }
    assert len(source_entity_ids) == 2
    llm_call = next(record for record in tracer.records if record.kind == EventKind.LLM_CALL.value)
    assert llm_call.payload.get("trusted_citation_ids", {}) == {}
    response = next(
        record for record in tracer.records if record.kind == EventKind.LLM_RESPONSE.value
    )
    response_entity_id = next(
        identifier
        for identifier, entity in provenance["entity"].items()
        if entity.get("sk:eventIdHash") == source_id_hash(response.event_id)
    )
    response_activity = next(
        activity
        for activity in provenance["activity"].values()
        if activity.get("sk:eventIdHash") == source_id_hash(response.event_id)
    )

    assert response_activity.get("sk:lineageStatus") == "unresolved"
    assert not any(
        relation["prov:generatedEntity"] == response_entity_id
        and relation["prov:usedEntity"] in source_entity_ids
        for relation in provenance["wasDerivedFrom"].values()
    )
    assert llm_call.payload.get("unresolvable_citation_ids") == (locator,)
    assert "unresolvable_citation_ids" not in response.payload
    assert locator not in str(provenance)
    assert all("unresolvable_citation_ids" not in message for message in provider_messages)


async def test_rag_without_visual_backend_keeps_text_pipeline_path():
    rag = _make_rag()
    calls = []

    async def fake_ask(query):
        calls.append(query)
        return "text answer"

    rag._pipeline.ask = fake_ask

    assert await rag.ask("question") == "text answer"
    assert calls == ["question"]


def test_multimodal_rag_is_a_named_rag_facade():
    assert issubclass(MultimodalRAG, RAG)
    assert MultimodalRAG.__doc__
    assert TopLevelMultimodalRAG is MultimodalRAG


@pytest.mark.asyncio
async def test_rag_visual_stream_appends_citations_and_uses_anthropic_blocks():
    rag = _make_rag(visual_embeddings=FakeVisualEmbeddings())
    rag._pipeline.config.llm.config.provider = "anthropic"
    await rag.visual_retriever.add_pages(
        [VisualPage(image=b"page-a", text="answer", metadata={"page": 2, "bbox": [1, 2, 3, 4]})]
    )

    captured = {}

    async def stream_with_messages(messages, **kwargs):
        captured["messages"] = messages
        yield "partial"

    rag._pipeline.config.llm.stream_with_messages = stream_with_messages
    output = "".join([token async for token in rag.stream("answer")])

    content = captured["messages"][-1]["content"]
    assert any(block.get("type") == "image" for block in content)
    assert "partial" in output
    assert "page 2, bbox [1, 2, 3, 4]" in output


@pytest.mark.asyncio
async def test_rag_visual_ingestion_keeps_text_fallback_when_renderer_fails(tmp_path, monkeypatch):
    pdf_path = tmp_path / "broken.pdf"
    pdf_path.write_bytes(b"pdf")
    rag = _make_rag(
        visual_embeddings=FakeVisualEmbeddings(),
        visual_renderer=BrokenRenderer(),
    )

    async def fake_aload(self):
        return [Document(text="OCR fallback", metadata={"page": 1})]

    monkeypatch.setattr(PDFLoader, "aload", fake_aload)
    await rag.add_async(str(pdf_path))

    assert rag.visual_retriever is not None
    assert rag.visual_retriever.page_count == 0
    assert rag._pipeline.add_documents.await_count == 1


@pytest.mark.asyncio
async def test_rag_visual_context_packer_preserves_selected_page_image():
    packer = RecordingPacker()
    rag = _make_rag(visual_embeddings=FakeVisualEmbeddings(), context_packer=packer)
    await rag.visual_retriever.add_pages(
        [
            VisualPage(
                image=b"\xff\xd8\xffjpeg",
                text="JPEG evidence",
                metadata={"page": 3, "bbox": "full-page"},
            )
        ]
    )

    captured = {}

    async def generate_with_messages(messages, **kwargs):
        captured["messages"] = messages
        return "answer"

    rag._pipeline.config.llm.generate_with_messages = generate_with_messages
    await rag.ask("Which page?")

    assert packer.calls and packer.calls[0][1] == "Which page?"
    content = captured["messages"][-1]["content"]
    image_block = next(block for block in content if block.get("type") == "image_url")
    assert image_block["image_url"]["url"].startswith("data:image/jpeg;base64,")
    prompt_text = content[0]["text"]
    assert "JPEG evidence" in prompt_text


@pytest.mark.asyncio
async def test_rag_visual_text_only_provider_does_not_stringify_image_blocks():
    rag = _make_rag(visual_embeddings=FakeVisualEmbeddings())
    rag._pipeline.config.llm.supports_multimodal = False
    await rag.visual_retriever.add_pages(
        [VisualPage(image=b"page-a", text="OCR evidence", metadata={"page": 1})]
    )

    captured = {}

    async def generate_with_messages(messages, **kwargs):
        captured["messages"] = messages
        return "text answer"

    rag._pipeline.config.llm.generate_with_messages = generate_with_messages
    await rag.ask("What is shown?")

    content = captured["messages"][-1]["content"]
    assert isinstance(content, str)
    assert "image_url" not in content
    assert "does not accept image messages" in content


def test_rag_rejects_visual_backend_and_retriever_together():
    visual_retriever = VisualDocumentRetriever(FakeVisualEmbeddings())
    with pytest.raises(ValueError, match="either visual_retriever"):
        RAG(
            model="gpt-4o-mini",
            api_key="[REDACTED]",
            visual_embeddings=FakeVisualEmbeddings(),
            visual_retriever=visual_retriever,
        )


def test_rag_rejects_visual_model_and_backend_together():
    with pytest.raises(ValueError, match="either visual_embeddings or visual_model"):
        RAG(
            model="gpt-4o-mini",
            api_key="[REDACTED]",
            visual_embeddings=FakeVisualEmbeddings(),
            visual_model="vidore/colpali-v1.3-hf",
        )
