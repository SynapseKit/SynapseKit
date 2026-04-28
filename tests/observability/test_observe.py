from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest

import synapsekit.observe as observe
from synapsekit.agents.base import BaseTool, ToolResult
from synapsekit.agents.function_calling import FunctionCallingAgent
from synapsekit.graph.graph import StateGraph
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


class TestObserveRagAgentGraph:
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

    @pytest.mark.asyncio
    async def test_function_calling_agent_emits_step_and_tool_spans(self):
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
        spans = observe.get_exporter().export_dicts()
        assert len(spans) == 1
        assert spans[0]["name"] == "agent.run"
        child_names = [child["name"] for child in spans[0]["children"]]
        assert child_names.count("agent.step") == 2
        assert "agent.final_answer" in child_names
        first_step = next(child for child in spans[0]["children"] if child["name"] == "agent.step")
        first_step_child_names = [child["name"] for child in first_step["children"]]
        assert "tool.call" in first_step_child_names
        assert "llm.generate" in first_step_child_names

    @pytest.mark.asyncio
    async def test_compiled_graph_emits_wave_and_node_spans(self):
        graph = StateGraph()
        graph.add_node("first", lambda state: {"first": True})
        graph.add_node("second", lambda state: {"second": state.get("first", False)})
        graph.add_edge("first", "second")
        graph.set_entry_point("first").set_finish_point("second")
        compiled = graph.compile()

        result = await compiled.run({})

        assert result["first"] is True
        assert result["second"] is True
        spans = observe.get_exporter().export_dicts()
        assert len(spans) == 1
        assert spans[0]["name"] == "graph.run"
        child_names = [child["name"] for child in spans[0]["children"]]
        assert "graph.wave" in child_names
        assert "graph.node" in child_names
