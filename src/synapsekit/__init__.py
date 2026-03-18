"""
SynapseKit — lightweight, async-first RAG framework.

3-line happy path:

    from synapsekit import RAG

    rag = RAG(model="gpt-4o-mini", api_key="sk-...")
    rag.add("Your document text here")

    async for token in rag.stream("What is the main topic?"):
        print(token, end="", flush=True)
"""

from __future__ import annotations

from .agents import (
    AgentConfig,
    AgentExecutor,
    AgentMemory,
    AgentStep,
    ArxivSearchTool,
    BaseTool,
    BraveSearchTool,
    CalculatorTool,
    Crew,
    CrewAgent,
    CrewResult,
    DateTimeTool,
    DuckDuckGoSearchTool,
    EmailTool,
    FileListTool,
    FileReadTool,
    FileWriteTool,
    FunctionCallingAgent,
    GitHubAPITool,
    GraphQLTool,
    Handoff,
    HandoffChain,
    HandoffResult,
    HTTPRequestTool,
    HumanInputTool,
    JiraTool,
    JSONQueryTool,
    PDFReaderTool,
    PubMedSearchTool,
    PythonREPLTool,
    ReActAgent,
    RegexTool,
    SentimentAnalysisTool,
    ShellTool,
    SlackTool,
    SQLQueryTool,
    SQLSchemaInspectionTool,
    SummarizationTool,
    SupervisorAgent,
    Task,
    TavilySearchTool,
    ToolRegistry,
    ToolResult,
    TranslationTool,
    VectorSearchTool,
    WebScraperTool,
    WebSearchTool,
    WikipediaTool,
    WorkerAgent,
    YouTubeSearchTool,
    tool,
)
from .embeddings.backend import SynapsekitEmbeddings
from .graph import (
    END,
    BaseCheckpointer,
    CompiledGraph,
    ConditionalEdge,
    ConditionFn,
    Edge,
    EventHooks,
    ExecutionTrace,
    GraphConfigError,
    GraphEvent,
    GraphInterrupt,
    GraphRuntimeError,
    GraphState,
    InMemoryCheckpointer,
    InterruptState,
    JSONFileCheckpointer,
    Node,
    NodeFn,
    SQLiteCheckpointer,
    StateField,
    StateGraph,
    TraceEntry,
    TypedState,
    agent_node,
    approval_node,
    dynamic_route_node,
    fan_out_node,
    llm_node,
    rag_node,
    sse_stream,
    subgraph_node,
    ws_stream,
)
from .llm.base import BaseLLM, LLMConfig
from .llm.structured import generate_structured
from .loaders.base import Document
from .loaders.csv import CSVLoader
from .loaders.directory import DirectoryLoader
from .loaders.html import HTMLLoader
from .loaders.json_loader import JSONLoader
from .loaders.markdown import MarkdownLoader
from .loaders.pdf import PDFLoader
from .loaders.text import StringLoader, TextLoader
from .loaders.web import WebLoader
from .mcp import MCPClient, MCPServer, MCPToolAdapter
from .memory.buffer import BufferMemory
from .memory.conversation import ConversationMemory
from .memory.entity import EntityMemory
from .memory.hybrid import HybridMemory
from .memory.sqlite import SQLiteConversationMemory
from .memory.summary_buffer import SummaryBufferMemory
from .memory.token_buffer import TokenBufferMemory
from .observability.tracer import TokenTracer
from .parsers.json_parser import JSONParser
from .parsers.list_parser import ListParser
from .parsers.pydantic_parser import PydanticParser
from .prompts.template import ChatPromptTemplate, FewShotPromptTemplate, PromptTemplate
from .rag.facade import RAG
from .rag.pipeline import RAGConfig, RAGPipeline
from .retrieval.adaptive import AdaptiveRAGRetriever
from .retrieval.base import VectorStore
from .retrieval.cohere_reranker import CohereReranker
from .retrieval.contextual import ContextualRetriever
from .retrieval.contextual_compression import ContextualCompressionRetriever
from .retrieval.crag import CRAGRetriever
from .retrieval.cross_encoder import CrossEncoderReranker
from .retrieval.ensemble import EnsembleRetriever
from .retrieval.flare import FLARERetriever
from .retrieval.hybrid_search import HybridSearchRetriever
from .retrieval.hyde import HyDERetriever
from .retrieval.multi_step import MultiStepRetriever
from .retrieval.parent_document import ParentDocumentRetriever
from .retrieval.query_decomposition import QueryDecompositionRetriever
from .retrieval.rag_fusion import RAGFusionRetriever
from .retrieval.retriever import Retriever
from .retrieval.self_query import SelfQueryRetriever
from .retrieval.self_rag import SelfRAGRetriever
from .retrieval.sentence_window import SentenceWindowRetriever
from .retrieval.step_back import StepBackRetriever
from .retrieval.vectorstore import InMemoryVectorStore
from .text_splitters import (
    BaseSplitter,
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SemanticSplitter,
    TokenAwareSplitter,
)

__version__ = "0.7.0"
__all__ = [
    # Facade
    "RAG",
    # Pipeline
    "RAGPipeline",
    "RAGConfig",
    # LLM
    "BaseLLM",
    "LLMConfig",
    "AzureOpenAILLM",
    "CerebrasLLM",
    "DeepSeekLLM",
    "FireworksLLM",
    "GroqLLM",
    "OpenRouterLLM",
    "PerplexityLLM",
    "TogetherLLM",
    # Embeddings
    "SynapsekitEmbeddings",
    # Vector stores
    "VectorStore",
    "InMemoryVectorStore",
    "ChromaVectorStore",
    "FAISSVectorStore",
    "QdrantVectorStore",
    "PineconeVectorStore",
    # Retrieval
    "Retriever",
    "AdaptiveRAGRetriever",
    "CohereReranker",
    "HybridSearchRetriever",
    "MultiStepRetriever",
    "RAGFusionRetriever",
    "ContextualRetriever",
    "ContextualCompressionRetriever",
    "CRAGRetriever",
    "CrossEncoderReranker",
    "EnsembleRetriever",
    "FLARERetriever",
    "HyDERetriever",
    "ParentDocumentRetriever",
    "QueryDecompositionRetriever",
    "SelfQueryRetriever",
    "SelfRAGRetriever",
    "SentenceWindowRetriever",
    "StepBackRetriever",
    # Memory / observability
    "BufferMemory",
    "ConversationMemory",
    "EntityMemory",
    "HybridMemory",
    "SQLiteConversationMemory",
    "SummaryBufferMemory",
    "TokenBufferMemory",
    "TokenTracer",
    # Loaders
    "Document",
    "TextLoader",
    "StringLoader",
    "PDFLoader",
    "HTMLLoader",
    "CSVLoader",
    "JSONLoader",
    "DirectoryLoader",
    "DocxLoader",
    "MarkdownLoader",
    "WebLoader",
    "ExcelLoader",
    "PowerPointLoader",
    # Parsers
    "JSONParser",
    "PydanticParser",
    "ListParser",
    # Prompts
    "PromptTemplate",
    "ChatPromptTemplate",
    "FewShotPromptTemplate",
    # Agents
    "BaseTool",
    "ToolResult",
    "ToolRegistry",
    "AgentMemory",
    "AgentStep",
    "ReActAgent",
    "FunctionCallingAgent",
    "AgentExecutor",
    "AgentConfig",
    # Tool decorator
    "tool",
    # Multi-agent
    "Crew",
    "CrewAgent",
    "CrewResult",
    "Handoff",
    "HandoffChain",
    "HandoffResult",
    "SupervisorAgent",
    "Task",
    "WorkerAgent",
    # MCP
    "MCPClient",
    "MCPServer",
    "MCPToolAdapter",
    # Built-in tools
    "ArxivSearchTool",
    "BraveSearchTool",
    "CalculatorTool",
    "DateTimeTool",
    "DuckDuckGoSearchTool",
    "FileListTool",
    "FileReadTool",
    "FileWriteTool",
    "GraphQLTool",
    "HTTPRequestTool",
    "HumanInputTool",
    "JiraTool",
    "JSONQueryTool",
    "PDFReaderTool",
    "PythonREPLTool",
    "RegexTool",
    "SentimentAnalysisTool",
    "ShellTool",
    "SlackTool",
    "SQLQueryTool",
    "SQLSchemaInspectionTool",
    "SummarizationTool",
    "TavilySearchTool",
    "TranslationTool",
    "EmailTool",
    "GitHubAPITool",
    "PubMedSearchTool",
    "WebScraperTool",
    "WebSearchTool",
    "VectorSearchTool",
    "WikipediaTool",
    "YouTubeSearchTool",
    # Text splitters
    "BaseSplitter",
    "CharacterTextSplitter",
    "RecursiveCharacterTextSplitter",
    "TokenAwareSplitter",
    "SemanticSplitter",
    # Graph workflows
    "END",
    "GraphState",
    "GraphConfigError",
    "GraphRuntimeError",
    "Node",
    "NodeFn",
    "agent_node",
    "approval_node",
    "dynamic_route_node",
    "llm_node",
    "rag_node",
    "subgraph_node",
    "GraphInterrupt",
    "InterruptState",
    "Edge",
    "ConditionalEdge",
    "ConditionFn",
    "EventHooks",
    "GraphEvent",
    "StateField",
    "StateGraph",
    "TypedState",
    "CompiledGraph",
    "ExecutionTrace",
    "TraceEntry",
    "fan_out_node",
    "sse_stream",
    "ws_stream",
    # Checkpointers
    "BaseCheckpointer",
    "InMemoryCheckpointer",
    "JSONFileCheckpointer",
    "SQLiteCheckpointer",
    # Structured output
    "generate_structured",
]

# Lazy imports for optional backends
_LAZY_IMPORTS = {
    # Vector stores
    "ChromaVectorStore": "retrieval.chroma",
    "FAISSVectorStore": "retrieval.faiss",
    "QdrantVectorStore": "retrieval.qdrant",
    "PineconeVectorStore": "retrieval.pinecone",
    # LLM providers
    "AzureOpenAILLM": "llm.azure_openai",
    "CerebrasLLM": "llm.cerebras",
    "DeepSeekLLM": "llm.deepseek",
    "FireworksLLM": "llm.fireworks",
    "GroqLLM": "llm.groq",
    "OpenRouterLLM": "llm.openrouter",
    "PerplexityLLM": "llm.perplexity",
    "TogetherLLM": "llm.together",
    # Loaders
    "DocxLoader": "loaders.docx",
    "ExcelLoader": "loaders.excel",
    "PowerPointLoader": "loaders.pptx",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        mod = importlib.import_module(f".{_LAZY_IMPORTS[name]}", __name__)
        cls = getattr(mod, name)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
