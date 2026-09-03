from .adaptive import AdaptiveRAGRetriever
from .agentic_rag import AgenticRAGRetriever
from .base import VectorStore
from .cohere_reranker import CohereReranker
from .context_packer import ContextPacker
from .contextual_compression import ContextualCompressionRetriever
from .crag import CRAGRetriever
from .cross_encoder import CrossEncoderReranker
from .document_augmentation import DocumentAugmentationRetriever
from .ensemble import EnsembleRetriever
from .federated import FederatedRetriever
from .flare import FLARERetriever
from .full_context import FullContextRetriever
from .graphrag import GraphRAGRetriever, KnowledgeGraph
from .hybrid_search import HybridSearchRetriever
from .hyde import HyDERetriever
from .jina_reranker import JinaReranker
from .late_chunking import LateChunkingRetriever
from .mixedbread_reranker import MixedbreadReranker
from .mongodb_atlas import MongoDBAtlasVectorStore
from .multi_step import MultiStepRetriever
from .parent_document import ParentDocumentRetriever
from .property_graph import (
    ExtractedEntity,
    ExtractedRelationship,
    GraphVectorStore,
    KnowledgeGraphExtraction,
    KnowledgeGraphExtractor,
    Neo4jPropertyGraphBackend,
    NetworkXPropertyGraphBackend,
    PropertyGraphEdge,
    PropertyGraphNode,
)
from .query_decomposition import QueryDecompositionRetriever
from .raptor import RAPTORRetriever
from .reranker import Reranker
from .retriever import Retriever
from .self_query import SelfQueryRetriever
from .self_rag import SelfRAGRetriever
from .step_back import StepBackRetriever
from .strategies.colbert import ColBERTRetriever
from .token_counting import TokenCounter
from .vectorstore import InMemoryVectorStore
from .voyage_reranker import VoyageReranker
from .world_model import (
    CausalLinker,
    EntityMention,
    EntityResolver,
    EventMention,
    ExtractionPolicy,
    ExtractionResult,
    GraphQueryResult,
    HeuristicWorldModelExtractor,
    HybridWorldModelRetriever,
    InMemoryWorldGraphBackend,
    KuzuWorldGraphBackend,
    LLMWorldModelExtractor,
    Neo4jWorldGraphBackend,
    RelationMention,
    WorldModelEdge,
    WorldModelNode,
    WorldModelQueryResult,
    WorldModelRAG,
)

__all__ = [
    "AdaptiveRAGRetriever",
    "AgenticRAGRetriever",
    "AzureAISearchVectorStore",
    "DocumentAugmentationRetriever",
    "CouchbaseVectorStore",
    "DeepLakeVectorStore",
    "LateChunkingRetriever",
    "RAPTORRetriever",
    "SingleStoreVectorStore",
    "SurrealDBVectorStore",
    "TiDBVectorStore",
    "TurbopufferVectorStore",
    "VertexAIVectorStore",
    "MyScaleVectorStore",
    "CassandraVectorStore",
    "ChromaVectorStore",
    "ClickHouseVectorStore",
    "CohereReranker",
    "ColBERTRetriever",
    "ContextPacker",
    "ContextualCompressionRetriever",
    "CRAGRetriever",
    "CrossEncoderReranker",
    "DuckDBVectorStore",
    "ElasticsearchVectorStore",
    "EnsembleRetriever",
    "FederatedRetriever",
    "FAISSVectorStore",
    "FLARERetriever",
    "FullContextRetriever",
    "GraphRAGRetriever",
    "GraphVectorStore",
    "HybridSearchRetriever",
    "HyDERetriever",
    "InMemoryVectorStore",
    "JinaReranker",
    "KnowledgeGraph",
    "KnowledgeGraphExtraction",
    "KnowledgeGraphExtractor",
    "LanceDBVectorStore",
    "MarqoVectorStore",
    "MilvusVectorStore",
    "MongoDBAtlasVectorStore",
    "MixedbreadReranker",
    "MultiStepRetriever",
    "Neo4jPropertyGraphBackend",
    "NetworkXPropertyGraphBackend",
    "OpenSearchVectorStore",
    "ParentDocumentRetriever",
    "PropertyGraphEdge",
    "PropertyGraphNode",
    "PGVectorStore",
    "PineconeVectorStore",
    "QdrantVectorStore",
    "QueryDecompositionRetriever",
    "RedisVectorStore",
    "Reranker",
    "Retriever",
    "SelfQueryRetriever",
    "SelfRAGRetriever",
    "SQLiteVecStore",
    "StepBackRetriever",
    "SupabaseVectorStore",
    "TokenCounter",
    "TypesenseVectorStore",
    "VectorStore",
    "VespaVectorStore",
    "VoyageReranker",
    "WeaviateVectorStore",
    "CausalLinker",
    "EntityMention",
    "EntityResolver",
    "EventMention",
    "ExtractionPolicy",
    "ExtractionResult",
    "GraphQueryResult",
    "HeuristicWorldModelExtractor",
    "HybridWorldModelRetriever",
    "InMemoryWorldGraphBackend",
    "KuzuWorldGraphBackend",
    "LLMWorldModelExtractor",
    "Neo4jWorldGraphBackend",
    "RelationMention",
    "WorldModelEdge",
    "WorldModelNode",
    "WorldModelRAG",
    "WorldModelQueryResult",
    "ExtractedEntity",
    "ExtractedRelationship",
    "ZillizVectorStore",
]

_BACKENDS = {
    "AzureAISearchVectorStore": ".azure_ai_search",
    "CassandraVectorStore": ".cassandra_vector",
    "ChromaVectorStore": ".chroma",
    "ClickHouseVectorStore": ".clickhouse_vector",
    "CouchbaseVectorStore": ".couchbase_vector",
    "DeepLakeVectorStore": ".deeplake",
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
    "SingleStoreVectorStore": ".singlestore_vector",
    "SQLiteVecStore": ".sqlite_vec",
    "SurrealDBVectorStore": ".surrealdb_vector",
    "SupabaseVectorStore": ".supabase_vector",
    "TiDBVectorStore": ".tidb_vector",
    "TurbopufferVectorStore": ".turbopuffer",
    "TypesenseVectorStore": ".typesense_vector",
    "VertexAIVectorStore": ".vertex_ai_vector",
    "VespaVectorStore": ".vespa",
    "WeaviateVectorStore": ".weaviate",
    "ZillizVectorStore": ".zilliz_vector",
    "MyScaleVectorStore": ".myscale_vector",
}


def __getattr__(name: str):
    if name in _BACKENDS:
        import importlib

        mod = importlib.import_module(_BACKENDS[name], __name__)
        cls = getattr(mod, name)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
