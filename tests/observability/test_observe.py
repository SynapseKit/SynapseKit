from __future__ import annotations

from collections.abc import AsyncGenerator

import numpy as np
import pytest

import synapsekit.observe as observe
from synapsekit.agents.base import BaseTool, ToolResult
from synapsekit.agents.function_calling import FunctionCallingAgent
from synapsekit.agents.react import ReActAgent
from synapsekit.embeddings.backend import SynapsekitEmbeddings
from synapsekit.graph.graph import StateGraph
from synapsekit.llm.base import BaseLLM, LLMConfig
from synapsekit.memory.conversation import ConversationMemory
from synapsekit.observability.cost_tracker import CostTracker
from synapsekit.rag.pipeline import RAGConfig, RAGPipeline
from synapsekit.retrieval.retriever import Retriever
from synapsekit.retrieval.vectorstore import InMemoryVectorStore


class StreamingLLM(BaseLLM):
    def __init__(self, *, responses: list[str] | None = None) -> None:
        super().__init__(LLMConfig(model="gpt-4o-mini", api_key="test", provider="openai"))
        self._responses = list(responses or ["ok"])

    async def stream(self, prompt: str, **kw) -> AsyncGenerator[str, None]:
        self._input_tokens += 3
        self._output_tokens += 2
        for chunk in self._responses:
            yield chunk

    async def stream_with_messages(
        self, messages: list[dict[str, str]], **kw
    ) -> AsyncGenerator[str, None]:
        self._input_tokens += 8
        self._output_tokens += 4
        for chunk in self._responses:
            yield chunk


class ToolCallingLLM(BaseLLM):
    def __init__(self, responses: list[dict[str, object]]) -> None:
        super().__init__(LLMConfig(model="gpt-4o-mini", api_key="test", provider="openai"))
        self._responses = list(responses)

    async def stream(self, prompt: str, **kw) -> AsyncGenerator[str, None]:
        yield "unused"

    async def _call_with_tools_impl(self, messages, tools):
        self._input_tokens += 5
        self._output_tokens += 3
        return self._responses.pop(0)


class ReactLoopLLM(BaseLLM):
    def __init__(self) -> None:
        super().__init__(LLMConfig(model="gpt-4o-mini", api_key="test", provider="openai"))
        self._responses = [
            "Thought: I should use a tool.\nAction: echo\nAction Input: hello",
            "Final Answer: done",
        ]

    async def stream(self, prompt: str, **kw) -> AsyncGenerator[str, None]:
        self._input_tokens += 4
        self._output_tokens += 2
        yield self._responses.pop(0)


class AddTool(BaseTool):
    name = "add"
    description = "Add two numbers."
    parameters = {
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "required": ["a", "b"],
    }

    async def run(self, a=0, b=0, **kwargs) -> ToolResult:
        return ToolResult(output=str(a + b))


class EchoTool(BaseTool):
    name = "echo"
    description = "Echo the provided text."
    parameters = {
        "type": "object",
        "properties": {"input": {"type": "string"}},
        "required": ["input"],
    }

    async def run(self, input: str = "", **kwargs) -> ToolResult:
        return ToolResult(output=input)


class FakeEmbeddings(SynapsekitEmbeddings):
    def __init__(self) -> None:
        super().__init__(model="fake-embeddings")

    async def embed(self, texts: list[str]) -> np.ndarray:
        from synapsekit.observe.runtime import end_span, record_exception, start_span

        span = start_span(
            "embedding.encode",
            {
                "embedding.model": self.model,
                "embedding.batch_size": len(texts),
                "embedding.inputs": list(texts),
            },
        )
        try:
            rows = []
            for text in texts:
                lowered = text.lower()
                rows.append(
                    np.array(
                        [
                            2.0 if "python" in lowered else 0.2,
                            2.0 if "rag" in lowered else 0.1,
                            float(len(lowered.split()) or 1),
                        ],
                        dtype=np.float32,
                    )
                )
            arr = np.vstack(rows)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            return arr / norms
        except Exception as exc:
            record_exception(span, exc)
            raise
        finally:
            end_span(span)

    async def embed_one(self, text: str) -> np.ndarray:
        return (await self.embed([text]))[0]


@pytest.fixture(autouse=True)
def _reset_observe_state():
    observe.reset()
    observe.configure()
    observe.clear_exported_spans()
    yield
    observe.reset()


def _root_span() -> dict:
    spans = observe.get_exporter().export_dicts()
    assert len(spans) == 1
    return spans[0]


class TestObserveConfigAndPrivacy:
    def test_configure_supports_issue_exporters(self):
        for kind in ["console", "otlp", "jaeger", "langfuse", "honeycomb"]:
            exporter = observe.configure(exporter=kind, endpoint="http://localhost:4317")
            assert exporter.kind == kind
            assert exporter.endpoint == "http://localhost:4317"

    @pytest.mark.asyncio
    async def test_trace_llm_inputs_false_suppresses_prompt(self):
        observe.configure(trace_llm_inputs=False)
        llm = StreamingLLM(responses=["ok"])

        await llm.generate("top secret prompt")

        attrs = _root_span()["attributes"]
        assert "llm.input" not in attrs

    @pytest.mark.asyncio
    async def test_trace_llm_outputs_false_suppresses_completion(self):
        observe.configure(trace_llm_outputs=False)
        llm = StreamingLLM(responses=["classified"])

        await llm.generate("hello")

        attrs = _root_span()["attributes"]
        assert "llm.output" not in attrs

    def test_redact_keys_removes_sensitive_values(self):
        observe.configure(redact_keys=["api_key", "password"])

        span = observe.start_span(
            "custom.redaction",
            {
                "api_key": "sk-test",
                "nested": {"password": "hidden", "safe": "ok"},
            },
        )
        observe.end_span(span)

        attrs = _root_span()["attributes"]
        assert attrs["api_key"] == "[REDACTED]"
        assert attrs["nested"]["password"] == "[REDACTED]"
        assert attrs["nested"]["safe"] == "ok"


class TestObserveLLM:
    @pytest.mark.asyncio
    async def test_llm_stream_emits_span_with_tokens_cost_latency_and_output(self):
        llm = StreamingLLM(responses=["Hello", " world"])

        answer = await llm.generate("Say hi")

        assert answer == "Hello world"
        attrs = _root_span()["attributes"]
        assert attrs["llm.model"] == "gpt-4o-mini"
        assert attrs["llm.prompt_tokens"] == 3
        assert attrs["llm.completion_tokens"] == 2
        assert attrs["llm.total_tokens"] == 5
        assert attrs["llm.cost_usd"] > 0
        assert attrs["llm.latency_ms"] >= 0
        assert attrs["llm.input"] == "Say hi"
        assert attrs["llm.output"] == "Hello world"

    @pytest.mark.asyncio
    async def test_cost_matches_cost_tracker(self):
        llm = StreamingLLM(responses=["priced"])

        await llm.generate("price this")

        attrs = _root_span()["attributes"]
        tracker = CostTracker()
        tracker.record(
            model="gpt-4o-mini",
            input_tokens=attrs["llm.prompt_tokens"],
            output_tokens=attrs["llm.completion_tokens"],
            latency_ms=attrs["llm.latency_ms"],
        )
        assert attrs["llm.cost_usd"] == round(tracker.total_cost_usd, 6)

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
        root = _root_span()
        assert root["name"] == "custom.work"
        assert root["children"][0]["name"] == "llm.generate"


class TestObserveRagAgentGraph:
    @pytest.mark.asyncio
    async def test_rag_pipeline_emits_nested_spans_with_retrieval_children(self):
        vectorstore = InMemoryVectorStore(FakeEmbeddings())
        retriever = Retriever(vectorstore, rerank=True)
        await retriever.add(
            [
                "Python RAG systems need observability.",
                "Cooking recipes are not relevant.",
                "RAG pipelines benefit from tracing.",
            ]
        )
        observe.clear_exported_spans()
        llm = StreamingLLM(responses=["Answer"])
        pipeline = RAGPipeline(
            RAGConfig(
                llm=llm,
                retriever=retriever,
                memory=ConversationMemory(),
            )
        )

        answer = await pipeline.ask("python rag")

        assert answer == "Answer"
        root = _root_span()
        assert root["name"] == "rag.ask"
        child_names = [child["name"] for child in root["children"]]
        assert "rag.retrieve" in child_names
        assert "llm.generate" in child_names
        assert "rag.response" in child_names

        retrieve_span = next(child for child in root["children"] if child["name"] == "rag.retrieve")
        assert retrieve_span["attributes"]["rag.retrieved_chunks"] >= 1
        assert retrieve_span["attributes"]["rag.retrieval_latency_ms"] >= 0
        retrieve_children = [child["name"] for child in retrieve_span["children"]]
        assert "vector_store.search" in retrieve_children
        assert "reranker.rerank" in retrieve_children

        search_span = next(
            child for child in retrieve_span["children"] if child["name"] == "vector_store.search"
        )
        assert search_span["children"][0]["name"] == "embedding.encode"

    @pytest.mark.asyncio
    async def test_function_calling_agent_emits_step_tool_and_final_answer_spans(self):
        llm = ToolCallingLLM(
            [
                {
                    "content": None,
                    "tool_calls": [{"id": "t1", "name": "add", "arguments": {"a": 2, "b": 3}}],
                },
                {"content": "The answer is 5.", "tool_calls": None},
            ]
        )
        agent = FunctionCallingAgent(llm=llm, tools=[AddTool()])

        result = await agent.run("2 + 3?")

        assert result == "The answer is 5."
        root = _root_span()
        assert root["name"] == "agent.run"
        child_names = [child["name"] for child in root["children"]]
        assert child_names.count("agent.step") == 2
        assert "agent.final_answer" in child_names
        first_step = next(child for child in root["children"] if child["name"] == "agent.step")
        first_step_child_names = [child["name"] for child in first_step["children"]]
        assert "tool.call" in first_step_child_names
        assert "llm.generate" in first_step_child_names

    @pytest.mark.asyncio
    async def test_react_agent_emits_tool_span(self):
        agent = ReActAgent(llm=ReactLoopLLM(), tools=[EchoTool()])

        result = await agent.run("say hello")

        assert result == "done"
        root = _root_span()
        assert root["name"] == "agent.run"
        step_span = next(child for child in root["children"] if child["name"] == "agent.step")
        assert any(child["name"] == "tool.call" for child in step_span["children"])

    @pytest.mark.asyncio
    async def test_compiled_graph_emits_wave_and_node_spans_with_latency(self):
        graph = StateGraph()
        graph.add_node("first", lambda state: {"first": True})
        graph.add_node("second", lambda state: {"second": state.get("first", False)})
        graph.add_edge("first", "second")
        graph.set_entry_point("first").set_finish_point("second")
        compiled = graph.compile()

        result = await compiled.run({})

        assert result["first"] is True
        assert result["second"] is True
        root = _root_span()
        assert root["name"] == "graph.run"
        node_span = next(child for child in root["children"] if child["name"] == "graph.node")
        assert node_span["attributes"]["graph.node"] in {"first", "second"}
        assert node_span["duration_ms"] >= 0
