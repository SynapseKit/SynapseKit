from .base import VectorStore
from .cross_encoder import CrossEncoderReranker
from .parent_document import ParentDocumentRetriever
from .retriever import Retriever
from .self_query import SelfQueryRetriever
from .vectorstore import InMemoryVectorStore

__all__ = [
    "ChromaVectorStore",
    "CrossEncoderReranker",
    "FAISSVectorStore",
    "InMemoryVectorStore",
    "ParentDocumentRetriever",
    "PineconeVectorStore",
    "QdrantVectorStore",
    "Retriever",
    "SelfQueryRetriever",
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
