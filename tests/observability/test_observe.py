from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest

import synapsekit.observe as observe
from synapsekit.llm.base import BaseLLM, LLMConfig
from synapsekit.memory.conversation import ConversationMemory
from synapsekit.rag.pipeline import RAGConfig, RAGPipeline


class StreamingLLM(BaseLLM):
    def __init__(self, *, responses: list[str] | None = None) -> None:
        super().__init__(LLMConfig(model="gpt-4o-mini", api_key="test", provider="openai"))
        self._responses = list(responses or ["ok"])

    async def stream(self, prompt: str, **kw) -> AsyncGenerator[str, None]:
        self._input_tokens += 3
        self._output_tokens += 2
        for chunk in self._responses:
            yield chunk

    async def stream_with_messages(self, messages: list[dict[str, str]], **kw) -> AsyncGenerator[str, None]:
        self._input_tokens += 8
        self._output_tokens += 4
        for chunk in self._responses:
            yield chunk


class DummyRetriever:
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    async def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        return self._chunks[:top_k]

    async def add(self, chunks, meta):
        return None


@pytest.fixture(autouse=True)
def _reset_observe_state():
    observe.reset()
    observe.configure()
    observe.clear_exported_spans()
    yield
    observe.reset()


class TestObserveLLM:
    @pytest.mark.asyncio
    async def test_llm_stream_emits_span_with_tokens_and_output(self):
        llm = StreamingLLM(responses=["Hello", " world"])

        answer = await llm.generate("Say hi")

        assert answer == "Hello world"
        spans = observe.get_exporter().export_dicts()
        assert len(spans) == 1
        attrs = spans[0]["attributes"]
        assert spans[0]["name"] == "llm.generate"
        assert attrs["llm.model"] == "gpt-4o-mini"
        assert attrs["llm.prompt_tokens"] == 3
        assert attrs["llm.completion_tokens"] == 2
        assert attrs["llm.input"] == "Say hi"
        assert attrs["llm.output"] == "Hello world"

    @pytest.mark.asyncio
    async def test_sample_rate_zero_emits_no_spans(self):
        observe.configure(sample_rate=0.0)
        observe.clear_exported_spans()

        llm = StreamingLLM()
        await llm.generate("No trace")

        assert observe.get_exporter().export_dicts() == []

    @pytest.mark.asyncio
    async def test_trace_decorator_creates_child_span(self):
        @observe.trace("custom.work")
        async def do_work() -> str:
            llm = StreamingLLM(responses=["done"])
            return await llm.generate("decorated")

        result = await do_work()

        assert result == "done"
        spans = observe.get_exporter().export_dicts()
        assert len(spans) == 1
        assert spans[0]["name"] == "custom.work"
        assert spans[0]["children"][0]["name"] == "llm.generate"


class TestObserveRag:
    @pytest.mark.asyncio
    async def test_rag_pipeline_emits_nested_spans(self):
        llm = StreamingLLM(responses=["Answer"])
        pipeline = RAGPipeline(
            RAGConfig(
                llm=llm,
                retriever=DummyRetriever(["chunk one", "chunk two"]),
                memory=ConversationMemory(),
            )
        )

        answer = await pipeline.ask("What happened?")

        assert answer == "Answer"
        spans = observe.get_exporter().export_dicts()
        assert len(spans) == 1
        assert spans[0]["name"] == "rag.ask"
        child_names = [child["name"] for child in spans[0]["children"]]
        assert "rag.retrieve" in child_names
        assert "llm.generate" in child_names
        assert "rag.response" in child_names
