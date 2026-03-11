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

from ._compat import run_sync
from .embeddings.backend import SynapsekitEmbeddings
from .llm.base import BaseLLM, LLMConfig
from .loaders.base import Document
from .loaders.csv import CSVLoader
from .loaders.directory import DirectoryLoader
from .loaders.html import HTMLLoader
from .loaders.json_loader import JSONLoader
from .loaders.pdf import PDFLoader
from .loaders.text import StringLoader, TextLoader
from .loaders.web import WebLoader
from .memory.conversation import ConversationMemory
from .observability.tracer import TokenTracer
from .parsers.json_parser import JSONParser
from .parsers.list_parser import ListParser
from .parsers.pydantic_parser import PydanticParser
from .prompts.template import ChatPromptTemplate, FewShotPromptTemplate, PromptTemplate
from .rag.facade import RAG
from .rag.pipeline import RAGConfig, RAGPipeline
from .retrieval.base import VectorStore
from .retrieval.retriever import Retriever
from .retrieval.vectorstore import InMemoryVectorStore

from .agents import (
    AgentConfig,
    AgentExecutor,
    AgentMemory,
    AgentStep,
    BaseTool,
    CalculatorTool,
    FileReadTool,
    FunctionCallingAgent,
    PythonREPLTool,
    ReActAgent,
    SQLQueryTool,
    ToolRegistry,
    ToolResult,
    WebSearchTool,
)

from .graph import (
    END,
    CompiledGraph,
    ConditionalEdge,
    ConditionFn,
    Edge,
    GraphConfigError,
    GraphRuntimeError,
    GraphState,
    Node,
    NodeFn,
    StateGraph,
    agent_node,
    rag_node,
)

__version__ = "0.4.0"
__all__ = [
    # Facade
    "RAG",
    # Pipeline
    "RAGPipeline",
    "RAGConfig",
    # LLM
    "BaseLLM",
    "LLMConfig",
    # Embeddings
    "SynapsekitEmbeddings",
    # Vector stores
    "VectorStore",
    "InMemoryVectorStore",
    # Retrieval
    "Retriever",
    # Memory / observability
    "ConversationMemory",
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
    # Built-in tools
    "CalculatorTool",
    "FileReadTool",
    "PythonREPLTool",
    "SQLQueryTool",
    "WebSearchTool",
    # Graph workflows
    "END",
    "GraphState",
    "GraphConfigError",
    "GraphRuntimeError",
    "Node",
    "NodeFn",
    "agent_node",
    "rag_node",
    "Edge",
    "ConditionalEdge",
    "ConditionFn",
    "StateGraph",
    "CompiledGraph",
]
