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
    BaseTool,
    CalculatorTool,
    DateTimeTool,
    FileListTool,
    FileReadTool,
    FileWriteTool,
    FunctionCallingAgent,
    HTTPRequestTool,
    HumanInputTool,
    JSONQueryTool,
    PythonREPLTool,
    ReActAgent,
    RegexTool,
    SQLQueryTool,
    ToolRegistry,
    ToolResult,
    WebSearchTool,
    WikipediaTool,
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
    GraphConfigError,
    GraphInterrupt,
    GraphRuntimeError,
    GraphState,
    InMemoryCheckpointer,
    InterruptState,
    Node,
    NodeFn,
    SQLiteCheckpointer,
    StateGraph,
    agent_node,
    llm_node,
    rag_node,
    subgraph_node,
)
from .llm.base import BaseLLM, LLMConfig
from .llm.structured import generate_structured
from .loaders.base import Document
from .loaders.csv import CSVLoader
from .loaders.directory import DirectoryLoader
from .loaders.html import HTMLLoader
from .loaders.json_loader import JSONLoader
from .loaders.pdf import PDFLoader
from .loaders.text import StringLoader, TextLoader
from .loaders.web import WebLoader
from .memory.conversation import ConversationMemory
from .memory.hybrid import HybridMemory
from .memory.sqlite import SQLiteConversationMemory
from .memory.summary_buffer import SummaryBufferMemory
from .observability.tracer import TokenTracer
from .parsers.json_parser import JSONParser
from .parsers.list_parser import ListParser
from .parsers.pydantic_parser import PydanticParser
from .prompts.template import ChatPromptTemplate, FewShotPromptTemplate, PromptTemplate
from .rag.facade import RAG
from .rag.pipeline import RAGConfig, RAGPipeline
from .retrieval.base import VectorStore
from .retrieval.contextual import ContextualRetriever
from .retrieval.contextual_compression import ContextualCompressionRetriever
from .retrieval.crag import CRAGRetriever
from .retrieval.cross_encoder import CrossEncoderReranker
from .retrieval.ensemble import EnsembleRetriever
from .retrieval.parent_document import ParentDocumentRetriever
from .retrieval.query_decomposition import QueryDecompositionRetriever
from .retrieval.rag_fusion import RAGFusionRetriever
from .retrieval.retriever import Retriever
from .retrieval.self_query import SelfQueryRetriever
from .retrieval.sentence_window import SentenceWindowRetriever
from .retrieval.vectorstore import InMemoryVectorStore
from .text_splitters import (
    BaseSplitter,
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SemanticSplitter,
    TokenAwareSplitter,
)

__version__ = "0.6.2"
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
    "DeepSeekLLM",
    "FireworksLLM",
    "GroqLLM",
    "OpenRouterLLM",
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
    "RAGFusionRetriever",
    "ContextualRetriever",
    "ContextualCompressionRetriever",
    "CRAGRetriever",
    "CrossEncoderReranker",
    "EnsembleRetriever",
    "ParentDocumentRetriever",
    "QueryDecompositionRetriever",
    "SelfQueryRetriever",
    "SentenceWindowRetriever",
    # Memory / observability
    "ConversationMemory",
    "HybridMemory",
    "SQLiteConversationMemory",
    "SummaryBufferMemory",
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
    # Built-in tools
    "CalculatorTool",
    "DateTimeTool",
    "FileListTool",
    "FileReadTool",
    "FileWriteTool",
    "HTTPRequestTool",
    "HumanInputTool",
    "JSONQueryTool",
    "PythonREPLTool",
    "RegexTool",
    "SQLQueryTool",
    "WebSearchTool",
    "WikipediaTool",
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
    "llm_node",
    "rag_node",
    "subgraph_node",
    "GraphInterrupt",
    "InterruptState",
    "Edge",
    "ConditionalEdge",
    "ConditionFn",
    "StateGraph",
    "CompiledGraph",
    # Checkpointers
    "BaseCheckpointer",
    "InMemoryCheckpointer",
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
    "DeepSeekLLM": "llm.deepseek",
    "FireworksLLM": "llm.fireworks",
    "GroqLLM": "llm.groq",
    "OpenRouterLLM": "llm.openrouter",
    "TogetherLLM": "llm.together",
    # Loaders
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
