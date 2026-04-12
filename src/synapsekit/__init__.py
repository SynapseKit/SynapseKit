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

from ._api import deprecated, experimental, public_api
from .a2a import A2AClient, A2AMessage, A2AServer, A2ATask, AgentCard, TaskState
from .agents import (
    ActionEvent,
    AgentConfig,
    AgentExecutor,
    AgentMemory,
    AgentStep,
    ArxivSearchTool,
    BaseTool,
    BingSearchTool,
    BraveSearchTool,
    CalculatorTool,
    CodeInterpreterTool,
    ContentFilter,
    Crew,
    CrewAgent,
    CrewResult,
    DateTimeTool,
    DuckDuckGoSearchTool,
    EmailTool,
    ErrorEvent,
    FileListTool,
    FileReadTool,
    FileWriteTool,
    FinalAnswerEvent,
    FunctionCallingAgent,
    GitHubAPITool,
    GoogleSearchTool,
    GraphQLTool,
    GuardrailResult,
    Guardrails,
    Handoff,
    HandoffChain,
    HandoffResult,
    HTTPRequestTool,
    HumanInputTool,
    ImageAnalysisTool,
    JiraTool,
    JSONQueryTool,
    LinearTool,
    NewsTool,
    ObservationEvent,
    PDFReaderTool,
    PIIDetector,
    PIIRedactor,
    PubMedSearchTool,
    PythonREPLTool,
    ReActAgent,
    RedactionResult,
    RegexTool,
    SentimentAnalysisTool,
    ShellTool,
    SlackTool,
    SpeechToTextTool,
    SQLQueryTool,
    SQLSchemaInspectionTool,
    StepEvent,
    StripeTool,
    SummarizationTool,
    SupervisorAgent,
    Task,
    TavilySearchTool,
    TextToSpeechTool,
    ThoughtEvent,
    TokenEvent,
    ToolRegistry,
    ToolResult,
    TopicRestrictor,
    TranslationTool,
    TwilioTool,
    VectorSearchTool,
    WeatherTool,
    WebScraperTool,
    WebSearchTool,
    WikipediaTool,
    WolframAlphaTool,
    WorkerAgent,
    YouTubeSearchTool,
    tool,
)
from .embeddings.backend import SynapsekitEmbeddings
from .evaluation import (
    EvalCaseMeta,
    EvalRegression,
    EvalSnapshot,
    EvaluationPipeline,
    EvaluationResult,
    FaithfulnessMetric,
    GroundednessMetric,
    MetricDelta,
    MetricResult,
    RegressionReport,
    RelevancyMetric,
    eval_case,
)
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
    GraphVisualizer,
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
    get_mermaid_with_trace,
    llm_node,
    rag_node,
    sse_stream,
    subgraph_node,
    ws_stream,
)
from .llm.base import BaseLLM, LLMConfig
from .llm.cost_router import QUALITY_TABLE, CostRouter, CostRouterConfig, RouterModelSpec
from .llm.fallback_chain import FallbackChain, FallbackChainConfig
from .llm.multimodal import AudioContent, ImageContent, MultimodalMessage
from .llm.structured import generate_structured
from .loaders.arxiv import ArXivLoader
from .loaders.base import Document
from .loaders.confluence import ConfluenceLoader
from .loaders.csv import CSVLoader
from .loaders.directory import DirectoryLoader
from .loaders.email import EmailLoader
from .loaders.gcs import GCSLoader
from .loaders.git import GitLoader
from .loaders.github import GitHubLoader
from .loaders.google_sheets import GoogleSheetsLoader
from .loaders.html import HTMLLoader
from .loaders.image import ImageLoader
from .loaders.jira import JiraLoader
from .loaders.json_loader import JSONLoader
from .loaders.markdown import MarkdownLoader
from .loaders.mongodb import MongoDBLoader
from .loaders.pdf import PDFLoader
from .loaders.rss import RSSLoader
from .loaders.sql import SQLLoader
from .loaders.teams import TeamsLoader
from .loaders.text import StringLoader, TextLoader
from .loaders.web import WebLoader
from .loaders.wikipedia import WikipediaLoader
from .mcp import MCPClient, MCPServer, MCPToolAdapter
from .memory.buffer import BufferMemory
from .memory.conversation import ConversationMemory
from .memory.entity import EntityMemory
from .memory.hybrid import HybridMemory
from .memory.redis import RedisConversationMemory
from .memory.sqlite import SQLiteConversationMemory
from .memory.summary_buffer import SummaryBufferMemory
from .memory.token_buffer import TokenBufferMemory
from .observability import (
    AuditEntry,
    AuditLog,
    BudgetExceededError,
    BudgetGuard,
    BudgetLimit,
    CircuitState,
    CostRecord,
    CostTracker,
    DistributedTracer,
    OTelExporter,
    Span,
    TraceSpan,
    TracingMiddleware,
    TracingUI,
)
from .observability.tracer import TokenTracer
from .parsers.json_parser import JSONParser
from .parsers.list_parser import ListParser
from .parsers.pydantic_parser import PydanticParser
from .plugins import PluginRegistry
from .prompts.hub import PromptHub
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
from .retrieval.graphrag import GraphRAGRetriever, KnowledgeGraph
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
    CodeSplitter,
    HTMLTextSplitter,
    JSONSplitter,
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
    SemanticSplitter,
    SentenceTextSplitter,
    SentenceWindowSplitter,
    TokenAwareSplitter,
)

__version__ = "1.3.0"
__all__ = [
    # Facade
    "RAG",
    # Pipeline
    "RAGPipeline",
    "RAGConfig",
    # LLM
    "AsyncLRUCache",
    "DynamoDBCacheBackend",
    "MemcachedCacheBackend",
    "BaseLLM",
    "LLMConfig",
    "CostRouter",
    "CostRouterConfig",
    "RouterModelSpec",
    "QUALITY_TABLE",
    "FallbackChain",
    "FallbackChainConfig",
    "AlephAlphaLLM",
    "AzureOpenAILLM",
    "CerebrasLLM",
    "CloudflareLLM",
    "DeepSeekLLM",
    "FireworksLLM",
    "GroqLLM",
    "HuggingFaceLLM",
    "MinimaxLLM",
    "MoonshotLLM",
    "NovitaLLM",
    "OpenRouterLLM",
    "PerplexityLLM",
    "SambaNovaLLM",
    "TogetherLLM",
    "VertexAILLM",
    "XaiLLM",
    "WriterLLM",
    "ZhipuLLM",
    # Embeddings
    "SynapsekitEmbeddings",
    # Vector stores
    "VectorStore",
    "InMemoryVectorStore",
    "ChromaVectorStore",
    "FAISSVectorStore",
    "LanceDBVectorStore",
    "MilvusVectorStore",
    "PGVectorStore",
    "QdrantVectorStore",
    "PineconeVectorStore",
    "WeaviateVectorStore",
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
    "GraphRAGRetriever",
    "KnowledgeGraph",
    "StepBackRetriever",
    # Cost intelligence
    "CostTracker",
    "CostRecord",
    "BudgetGuard",
    "BudgetLimit",
    "BudgetExceededError",
    "CircuitState",
    # Memory / observability
    "BufferMemory",
    "ConversationMemory",
    "EntityMemory",
    "HybridMemory",
    "RedisConversationMemory",
    "SQLiteConversationMemory",
    "SummaryBufferMemory",
    "TokenBufferMemory",
    "TokenTracer",
    # Loaders
    "Document",
    "TextLoader",
    "ArXivLoader",
    "StringLoader",
    "PDFLoader",
    "HTMLLoader",
    "CSVLoader",
    "JSONLoader",
    "DirectoryLoader",
    "ConfluenceLoader",
    "DocxLoader",
    "EmailLoader",
    "GCSLoader",
    "GitHubLoader",
    "GitLoader",
    "GoogleSheetsLoader",
    "JiraLoader",
    "MarkdownLoader",
    "MongoDBLoader",
    "SQLLoader",
    "SupabaseLoader",
    "TeamsLoader",
    "WebLoader",
    "RSSLoader",
    "WikipediaLoader",
    "ExcelLoader",
    "PowerPointLoader",
    # Parsers
    "JSONParser",
    "PydanticParser",
    "ListParser",
    # Prompts
    "PromptHub",
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
    "BingSearchTool",
    "BraveSearchTool",
    "CalculatorTool",
    "CodeInterpreterTool",
    "DateTimeTool",
    "DuckDuckGoSearchTool",
    "FileListTool",
    "FileReadTool",
    "FileWriteTool",
    "GraphQLTool",
    "HTTPRequestTool",
    "HumanInputTool",
    "ImageAnalysisTool",
    "JiraTool",
    "JSONQueryTool",
    "LinearTool",
    "NewsTool",
    "PDFReaderTool",
    "PythonREPLTool",
    "RegexTool",
    "SentimentAnalysisTool",
    "ShellTool",
    "SlackTool",
    "SpeechToTextTool",
    "SQLQueryTool",
    "SQLSchemaInspectionTool",
    "SummarizationTool",
    "TavilySearchTool",
    "TextToSpeechTool",
    "TranslationTool",
    "TwilioTool",
    "EmailTool",
    "GitHubAPITool",
    "GoogleSearchTool",
    "PubMedSearchTool",
    "WebScraperTool",
    "WebSearchTool",
    "VectorSearchTool",
    "StripeTool",
    "WeatherTool",
    "WolframAlphaTool",
    "WikipediaTool",
    "YouTubeSearchTool",
    # Text splitters
    "BaseSplitter",
    "HTMLTextSplitter",
    "CharacterTextSplitter",
    "MarkdownTextSplitter",
    "RecursiveCharacterTextSplitter",
    "CodeSplitter",
    "JSONSplitter",
    "TokenAwareSplitter",
    "SemanticSplitter",
    "SentenceTextSplitter",
    "SentenceWindowSplitter",
    # Graph workflows
    "END",
    "GraphState",
    "GraphVisualizer",
    "get_mermaid_with_trace",
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
    "PostgresCheckpointer",
    "RedisCheckpointer",
    "SQLiteCheckpointer",
    # Structured output
    "generate_structured",
    # Evaluation
    "EvalCaseMeta",
    "EvalRegression",
    "EvalSnapshot",
    "EvaluationPipeline",
    "EvaluationResult",
    "FaithfulnessMetric",
    "GroundednessMetric",
    "MetricDelta",
    "MetricResult",
    "RegressionReport",
    "RelevancyMetric",
    "eval_case",
    # Observability
    "AuditEntry",
    "AuditLog",
    "DistributedTracer",
    "OTelExporter",
    "Span",
    "TraceSpan",
    "TracingMiddleware",
    "TracingUI",
    # A2A Protocol
    "A2AClient",
    "A2AMessage",
    "A2AServer",
    "A2ATask",
    "AgentCard",
    "TaskState",
    # Guardrails
    "ContentFilter",
    "Guardrails",
    "GuardrailResult",
    "PIIDetector",
    "PIIRedactor",
    "RedactionResult",
    "TopicRestrictor",
    # Step events
    "ActionEvent",
    "ErrorEvent",
    "FinalAnswerEvent",
    "ObservationEvent",
    "StepEvent",
    "ThoughtEvent",
    "TokenEvent",
    # Multimodal
    "AudioContent",
    "ImageContent",
    "MultimodalMessage",
    "AudioLoader",
    "VideoLoader",
    "ImageLoader",
    "YAMLLoader",
    "DiscordLoader",
    "XMLLoader",
    "GoogleDriveLoader",
    # Plugins
    "PluginRegistry",
    # API stability markers
    "deprecated",
    "experimental",
    "public_api",
]

# Lazy imports for optional backends
_LAZY_IMPORTS = {
    # Vector stores
    "ChromaVectorStore": "retrieval.chroma",
    "FAISSVectorStore": "retrieval.faiss",
    "LanceDBVectorStore": "retrieval.lancedb",
    "MilvusVectorStore": "retrieval.milvus",
    "PGVectorStore": "retrieval.pgvector",
    "QdrantVectorStore": "retrieval.qdrant",
    "PineconeVectorStore": "retrieval.pinecone",
    "WeaviateVectorStore": "retrieval.weaviate",
    # LLM providers
    "AsyncLRUCache": "llm._cache",
    "DynamoDBCacheBackend": "llm._cache_dynamodb",
    "MemcachedCacheBackend": "llm._cache_memcached",
    "AzureOpenAILLM": "llm.azure_openai",
    "CerebrasLLM": "llm.cerebras",
    "VertexAILLM": "llm.vertex_ai",
    "DeepSeekLLM": "llm.deepseek",
    "FireworksLLM": "llm.fireworks",
    "GroqLLM": "llm.groq",
    "HuggingFaceLLM": "llm.huggingface",
    "OpenRouterLLM": "llm.openrouter",
    "PerplexityLLM": "llm.perplexity",
    "SambaNovaLLM": "llm.sambanova",
    "TogetherLLM": "llm.together",
    "MinimaxLLM": "llm.minimax",
    "NovitaLLM": "llm.novita",
    "MoonshotLLM": "llm.moonshot",
    "AlephAlphaLLM": "llm.aleph_alpha",
    "XaiLLM": "llm.xai",
    "WriterLLM": "llm.writer",
    "ZhipuLLM": "llm.zhipu",
    "CloudflareLLM": "llm.cloudflare",
    # Checkpointers
    "RedisCheckpointer": "graph.checkpointers.redis",
    "PostgresCheckpointer": "graph.checkpointers.postgres",
    # Loaders
    "AudioLoader": "loaders.audio",
    "VideoLoader": "loaders.video",
    "DocxLoader": "loaders.docx",
    "ExcelLoader": "loaders.excel",
    "PowerPointLoader": "loaders.pptx",
    "TeamsLoader": "loaders.teams",
    "YAMLLoader": "loaders.yaml_loader",
    "DiscordLoader": "loaders.discord",
    "XMLLoader": "loaders.xml_loader",
    "GoogleDriveLoader": "loaders.google_drive",
    "MongoDBLoader": "loaders.mongodb",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        mod = importlib.import_module(f".{_LAZY_IMPORTS[name]}", __name__)
        cls = getattr(mod, name)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
