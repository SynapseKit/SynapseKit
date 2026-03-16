from .adaptive import AdaptiveRAGRetriever
from .base import VectorStore
from .cohere_reranker import CohereReranker
from .contextual_compression import ContextualCompressionRetriever
from .crag import CRAGRetriever
from .cross_encoder import CrossEncoderReranker
from .ensemble import EnsembleRetriever
from .flare import FLARERetriever
from .hybrid_search import HybridSearchRetriever
from .hyde import HyDERetriever
from .multi_step import MultiStepRetriever
from .parent_document import ParentDocumentRetriever
from .query_decomposition import QueryDecompositionRetriever
from .retriever import Retriever
from .self_query import SelfQueryRetriever
from .self_rag import SelfRAGRetriever
from .step_back import StepBackRetriever
from .vectorstore import InMemoryVectorStore

__all__ = [
    "AdaptiveRAGRetriever",
    "ChromaVectorStore",
    "CohereReranker",
    "ContextualCompressionRetriever",
    "CRAGRetriever",
    "CrossEncoderReranker",
    "EnsembleRetriever",
    "FAISSVectorStore",
    "FLARERetriever",
    "HybridSearchRetriever",
    "HyDERetriever",
    "InMemoryVectorStore",
    "MultiStepRetriever",
    "ParentDocumentRetriever",
    "PineconeVectorStore",
    "QdrantVectorStore",
    "QueryDecompositionRetriever",
    "Retriever",
    "SelfQueryRetriever",
    "SelfRAGRetriever",
    "StepBackRetriever",
    "VectorStore",
]

_BACKENDS = {
    "ChromaVectorStore": ".chroma",
    "FAISSVectorStore": ".faiss",
    "QdrantVectorStore": ".qdrant",
    "PineconeVectorStore": ".pinecone",
}


def __getattr__(name: str):
    if name in _BACKENDS:
        import importlib

        mod = importlib.import_module(_BACKENDS[name], __name__)
        cls = getattr(mod, name)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
