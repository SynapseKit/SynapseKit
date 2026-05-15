"""Knowledge Graph retrievers for multi-hop RAG."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..retriever import Retriever
    from .backends import BaseGraphStore
    from .builder import KnowledgeGraphBuilder


class KGRetriever:
    """Retrieves context by traversing a Knowledge Graph using depth-first search."""

    def __init__(
        self,
        store: BaseGraphStore,
        builder: KnowledgeGraphBuilder,
        max_hops: int = 2,
        min_confidence: float = 0.5,
    ) -> None:
        self._store = store
        self._builder = builder
        self._max_hops = max_hops
        self._min_confidence = min_confidence

    async def retrieve(self, query: str) -> list[str]:
        """Extract entities from the query and return associated document IDs."""
        entities = await self._builder.extract_entities(query)
        graph_doc_ids: list[str] = []

        for entity in entities:
            neighbors = self._store.get_neighbors(
                entity,
                max_hops=self._max_hops,
                min_confidence=self._min_confidence,
            )
            all_entities = {entity} | neighbors
            for ent in all_entities:
                graph_doc_ids.extend(self._store.get_related_documents(ent))

        # Deduplicate while preserving order
        return list(dict.fromkeys(graph_doc_ids))


class HybridKGRetriever:
    """Combines vector retrieval with Knowledge Graph traversal."""

    def __init__(self, vector_retriever: Retriever, kg_retriever: KGRetriever) -> None:
        self._vector_retriever = vector_retriever
        self._kg_retriever = kg_retriever

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[str]:
        """Perform hybrid retrieval: vector search + KG traversal."""
        # 1. Vector Search
        vector_results = await self._vector_retriever.retrieve(
            query, top_k=top_k, metadata_filter=metadata_filter
        )

        # 2. Graph Traversal (returns doc IDs)
        # Note: Since the KG store stores doc_ids but our vector retriever returns raw texts
        # by default, we need to map them or assume they store texts if the backend was
        # configured that way. In standard GraphRAG, the vector store stores the texts,
        # and the doc IDs are matching metadata.
        # For simplicity in this implementation, if the store uses texts as doc IDs, it merges them.
        graph_results = await self._kg_retriever.retrieve(query)

        # 3. Merge and deduplicate
        seen: set[str] = set()
        merged: list[str] = []

        for doc in vector_results + graph_results:
            if doc not in seen:
                seen.add(doc)
                merged.append(doc)

        return merged[:top_k]
