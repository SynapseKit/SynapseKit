from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class SpanAttributes:
    LLM_MODEL = "llm.model"
    LLM_PROVIDER = "llm.provider"
    LLM_INPUT = "llm.input"
    LLM_OUTPUT = "llm.output"
    LLM_PROMPT_TOKENS = "llm.prompt_tokens"
    LLM_COMPLETION_TOKENS = "llm.completion_tokens"
    LLM_TOTAL_TOKENS = "llm.total_tokens"
    LLM_COST_USD = "llm.cost_usd"
    LLM_LATENCY_MS = "llm.latency_ms"
    LLM_TOOL_CALLS = "llm.tool_calls"

    RAG_QUERY = "rag.query"
    RAG_TOP_K = "rag.top_k"
    RAG_RETRIEVED_CHUNKS = "rag.retrieved_chunks"
    RAG_TOP_SCORE = "rag.top_score"
    RAG_RETRIEVAL_LATENCY_MS = "rag.retrieval_latency_ms"
    RAG_RESPONSE_LENGTH = "rag.response_length"

    AGENT_TYPE = "agent.type"
    AGENT_STEP = "agent.step"
    AGENT_TOOL = "agent.tool"
    AGENT_TOOL_INPUT = "agent.tool_input"
    AGENT_TOOL_OUTPUT = "agent.tool_output"
    AGENT_TOOL_CALLS = "agent.tool_calls"
    AGENT_ANSWER_LENGTH = "agent.answer_length"
    AGENT_MAX_ITERATIONS = "agent.max_iterations"

    GRAPH_NODES = "graph.nodes"
    GRAPH_EDGES = "graph.edges"
    GRAPH_STEP = "graph.step"
    GRAPH_WAVE = "graph.wave"
    GRAPH_NODE = "graph.node"
    GRAPH_WAVE_COMPLETE = "graph.wave_complete"

    EMBEDDING_MODEL = "embedding.model"
    EMBEDDING_BATCH_SIZE = "embedding.batch_size"
    EMBEDDING_INPUTS = "embedding.inputs"

    VECTOR_STORE_TYPE = "vector_store.type"
    VECTOR_STORE_TOP_K = "vector_store.top_k"
    VECTOR_STORE_RESULTS = "vector_store.results"

    RERANKER_TYPE = "reranker.type"
    RERANKER_TOP_K = "reranker.top_k"
    RERANKER_CANDIDATES = "reranker.candidates"


@dataclass
class SpanBuilder:
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)

    def attr(self, key: str, value: Any) -> SpanBuilder:
        if value is not None:
            self.attributes[key] = value
        return self

    def extend(self, values: dict[str, Any]) -> SpanBuilder:
        for key, value in values.items():
            if value is not None:
                self.attributes[key] = value
        return self

    def build(self) -> tuple[str, dict[str, Any]]:
        return self.name, dict(self.attributes)
