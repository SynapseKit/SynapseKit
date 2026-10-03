from .agent_memory import AgentMemory
from .backends import (
    GraphAgentMemory,
    GraphMemoryBackend,
    InMemoryMemoryBackend,
    PostgresMemoryBackend,
    RedisMemoryBackend,
    SQLiteMemoryBackend,
)
from .base import BaseMemoryBackend, MemoryRecord
from .buffer import BufferMemory
from .conversation import ConversationMemory
from .diff_engine import DiffConflictError, FileDiffEngine
from .entity import EntityMemory
from .file_router import MemoryFileRouter
from .hybrid import HybridMemory
from .knowledge_graph_memory import KnowledgeGraphMemory
from .living_memory import LivingMemory
from .living_types import MemoryFileCategory, MemoryPatch, OccurrenceRecord, PatchStatus
from .patch_store import OccurrenceTracker, PatchStore
from .pii_filter import MemoryPIIFilter, PIIFilterResult
from .readonly_shared_memory import ReadOnlySharedMemory
from .redis import RedisConversationMemory
from .semantic_compressor import SemanticCompressorMemory
from .smart_context import SmartContextManager
from .sqlite import SQLiteConversationMemory
from .summary_buffer import SummaryBufferMemory
from .token_buffer import TokenBufferMemory
from .vector_memory import VectorConversationMemory

__all__ = [
    "AgentMemory",
    "BaseMemoryBackend",
    "MemoryRecord",
    "GraphAgentMemory",
    "GraphMemoryBackend",
    "InMemoryMemoryBackend",
    "SQLiteMemoryBackend",
    "RedisMemoryBackend",
    "PostgresMemoryBackend",
    "MongoDBMemoryBackend",
    "CassandraMemoryBackend",
    "ScyllaMemoryBackend",
    "ScyllaDBMemoryBackend",
    "DynamoDBMemoryBackend",
    "FirestoreMemoryBackend",
    "CosmosDBMemoryBackend",
    "BufferMemory",
    "ConversationMemory",
    "EntityMemory",
    "HybridMemory",
    "KnowledgeGraphMemory",
    "ReadOnlySharedMemory",
    "RedisConversationMemory",
    "SmartContextManager",
    "SQLiteConversationMemory",
    "SummaryBufferMemory",
    "SemanticCompressorMemory",
    "TokenBufferMemory",
    "VectorConversationMemory",
    "LivingMemory",
    "MemoryPatch",
    "OccurrenceRecord",
    "MemoryFileCategory",
    "PatchStatus",
    "FileDiffEngine",
    "DiffConflictError",
    "PatchStore",
    "OccurrenceTracker",
    "MemoryPIIFilter",
    "PIIFilterResult",
    "MemoryFileRouter",
]


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    _lazy = {
        "MongoDBMemoryBackend": "mongodb",
        "CassandraMemoryBackend": "cassandra",
        "ScyllaMemoryBackend": "cassandra",
        "ScyllaDBMemoryBackend": "scylla",
        "DynamoDBMemoryBackend": "dynamodb",
        "FirestoreMemoryBackend": "firestore",
        "CosmosDBMemoryBackend": "cosmos",
    }
    if name in _lazy:
        import importlib

        module = importlib.import_module(f".backends.{_lazy[name]}", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
