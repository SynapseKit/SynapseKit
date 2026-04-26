from .adaptive import AdaptiveRAGRetriever
from .agentic_rag import AgenticRAGRetriever
from .base import VectorStore
from .cohere_reranker import CohereReranker
from .contextual_compression import ContextualCompressionRetriever
from .crag import CRAGRetriever
from .cross_encoder import CrossEncoderReranker
from .document_augmentation import DocumentAugmentationRetriever
from .ensemble import EnsembleRetriever
from .flare import FLARERetriever
from .graphrag import GraphRAGRetriever, KnowledgeGraph
from .hybrid_search import HybridSearchRetriever
from .hyde import HyDERetriever
from .late_chunking import LateChunkingRetriever
from .mongodb_atlas import MongoDBAtlasVectorStore
from .multi_step import MultiStepRetriever
from .parent_document import ParentDocumentRetriever
from .query_decomposition import QueryDecompositionRetriever
from .raptor import RAPTORRetriever
from .retriever import Retriever
from .self_query import SelfQueryRetriever
from .self_rag import SelfRAGRetriever
from .step_back import StepBackRetriever
from .strategies.colbert import ColBERTRetriever
from .vectorstore import InMemoryVectorStore

__all__ = [
    "AdaptiveRAGRetriever",
    "AgenticRAGRetriever",
    "DocumentAugmentationRetriever",
    "LateChunkingRetriever",
    "RAPTORRetriever",
    "CassandraVectorStore",
    "ChromaVectorStore",
    "ClickHouseVectorStore",
    "CohereReranker",
    "ColBERTRetriever",
    "ContextualCompressionRetriever",
    "CRAGRetriever",
    "CrossEncoderReranker",
    "DuckDBVectorStore",
    "ElasticsearchVectorStore",
    "EnsembleRetriever",
    "FAISSVectorStore",
    "FLARERetriever",
    "GraphRAGRetriever",
    "HybridSearchRetriever",
    "HyDERetriever",
    "InMemoryVectorStore",
    "KnowledgeGraph",
    "LanceDBVectorStore",
    "MarqoVectorStore",
    "MilvusVectorStore",
    "MongoDBAtlasVectorStore",
    "MultiStepRetriever",
    "OpenSearchVectorStore",
    "ParentDocumentRetriever",
    "PGVectorStore",
    "PineconeVectorStore",
    "QdrantVectorStore",
    "QueryDecompositionRetriever",
    "RedisVectorStore",
    "Retriever",
    "SelfQueryRetriever",
    "SelfRAGRetriever",
    "SQLiteVecStore",
    "StepBackRetriever",
    "SupabaseVectorStore",
    "TypesenseVectorStore",
    "VectorStore",
    "VespaVectorStore",
    "WeaviateVectorStore",
    "ZillizVectorStore",
]

_BACKENDS = {
    "CassandraVectorStore": ".cassandra_vector",
    "ChromaVectorStore": ".chroma",
    "ClickHouseVectorStore": ".clickhouse_vector",
    "DuckDBVectorStore": ".duckdb_vector",
    "ElasticsearchVectorStore": ".elasticsearch_vector",
    "FAISSVectorStore": ".faiss",
    "LanceDBVectorStore": ".lancedb",
    "MarqoVectorStore": ".marqo_vector",
    "MilvusVectorStore": ".milvus",
    "OpenSearchVectorStore": ".opensearch_vector",
    "PGVectorStore": ".pgvector",
    "PineconeVectorStore": ".pinecone",
    "QdrantVectorStore": ".qdrant",
    "RedisVectorStore": ".redis_vector",
    "SQLiteVecStore": ".sqlite_vec",
    "SupabaseVectorStore": ".supabase_vector",
    "TypesenseVectorStore": ".typesense_vector",
    "VespaVectorStore": ".vespa",
    "WeaviateVectorStore": ".weaviate",
    "ZillizVectorStore": ".zilliz_vector",
}


def __getattr__(name: str):
    if name in _BACKENDS:
        import importlib

        mod = importlib.import_module(_BACKENDS[name], __name__)
        cls = getattr(mod, name)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
