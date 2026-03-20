"""GraphRAG: knowledge-graph-augmented retrieval with entity extraction."""

from __future__ import annotations

from collections import deque

from ..llm.base import BaseLLM
from .retriever import Retriever

_EXTRACT_ENTITIES_PROMPT = (
    "Extract the key entities (people, places, concepts, etc.) from the following text. "
    "Return only a comma-separated list of entities, nothing else.\n\n"
    "Text: {text}"
)

_EXTRACT_TRIPLES_PROMPT = (
    "Extract knowledge graph triples from the following text. "
    "Each triple should be in the format: subject|predicate|object\n"
    "Return one triple per line, nothing else.\n\n"
    "Text: {text}"
)


class KnowledgeGraph:
    """In-memory graph store for entity triples (subject, predicate, object).

    Usage::

        kg = KnowledgeGraph()
        kg.add_triple("Einstein", "developed", "general relativity")
        neighbors = kg.get_neighbors("Einstein", max_hops=2)
    """

    def __init__(self) -> None:
        self._adjacency: dict[str, set[str]] = {}
        self._triples: list[tuple[str, str, str]] = []
        self._entity_to_docs: dict[str, list[str]] = {}

    def add_triple(self, subject: str, predicate: str, obj: str) -> None:
        """Add a triple and update adjacency in both directions."""
        self._triples.append((subject, predicate, obj))
        self._adjacency.setdefault(subject, set()).add(obj)
        self._adjacency.setdefault(obj, set()).add(subject)

    def get_neighbors(self, entity: str, max_hops: int = 1) -> set[str]:
        """BFS traversal returning connected entities within *max_hops*."""
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(entity, 0)])

        while queue:
            current, depth = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            if depth < max_hops:
                for neighbor in self._adjacency.get(current, set()):
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))

        visited.discard(entity)
        return visited

    def get_related_documents(self, entity: str) -> list[str]:
        """Return document IDs linked to *entity*."""
        return list(self._entity_to_docs.get(entity, []))

    def add_document_link(self, entity: str, doc_id: str) -> None:
        """Link an entity to a document ID."""
        self._entity_to_docs.setdefault(entity, []).append(doc_id)

    async def build_from_documents(self, docs: list[str], llm: BaseLLM) -> None:
        """Use the LLM to extract entities and triples from *docs*."""
        for idx, doc in enumerate(docs):
            doc_id = f"doc_{idx}"

            # Extract triples
            triples_prompt = _EXTRACT_TRIPLES_PROMPT.format(text=doc)
            triples_response = await llm.generate(triples_prompt)

            for line in triples_response.strip().splitlines():
                parts = line.strip().split("|")
                if len(parts) == 3:
                    subject, predicate, obj = (p.strip() for p in parts)
                    self.add_triple(subject, predicate, obj)
                    self.add_document_link(subject, doc_id)
                    self.add_document_link(obj, doc_id)


class GraphRAGRetriever:
    """Knowledge-graph-augmented retriever that combines graph traversal
    with vector retrieval for improved results.

    Usage::

        graphrag = GraphRAGRetriever(retriever=retriever, llm=llm, knowledge_graph=kg)
        results = await graphrag.retrieve("What did Einstein discover?", top_k=5)
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: BaseLLM,
        knowledge_graph: KnowledgeGraph | None = None,
        max_hops: int = 2,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._knowledge_graph = knowledge_graph
        self._max_hops = max_hops

    async def _extract_entities(self, text: str) -> list[str]:
        """Use the LLM to extract entities from text."""
        prompt = _EXTRACT_ENTITIES_PROMPT.format(text=text)
        response = await self._llm.generate(prompt)
        return [e.strip() for e in response.split(",") if e.strip()]

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[str]:
        """Extract entities from the query, traverse the knowledge graph,
        merge graph-linked documents with vector retrieval, and deduplicate."""
        # Vector retrieval
        vector_results = await self._retriever.retrieve(
            query, top_k=top_k, metadata_filter=metadata_filter
        )

        if self._knowledge_graph is None:
            return vector_results[:top_k]

        # Entity extraction and graph traversal
        entities = await self._extract_entities(query)
        graph_doc_ids: list[str] = []
        for entity in entities:
            neighbors = self._knowledge_graph.get_neighbors(entity, max_hops=self._max_hops)
            all_entities = {entity} | neighbors
            for ent in all_entities:
                graph_doc_ids.extend(self._knowledge_graph.get_related_documents(ent))

        # Deduplicate while preserving order
        seen: set[str] = set()
        merged: list[str] = []
        for doc in vector_results + graph_doc_ids:
            if doc not in seen:
                seen.add(doc)
                merged.append(doc)

        return merged[:top_k]

    async def retrieve_with_graph(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> tuple[list[str], dict]:
        """Retrieve and return graph traversal metadata for transparency."""
        vector_results = await self._retriever.retrieve(
            query, top_k=top_k, metadata_filter=metadata_filter
        )

        if self._knowledge_graph is None:
            return vector_results[:top_k], {
                "entities_extracted": [],
                "graph_docs": [],
                "traversal_hops": self._max_hops,
            }

        entities = await self._extract_entities(query)
        graph_doc_ids: list[str] = []
        for entity in entities:
            neighbors = self._knowledge_graph.get_neighbors(entity, max_hops=self._max_hops)
            all_entities = {entity} | neighbors
            for ent in all_entities:
                graph_doc_ids.extend(self._knowledge_graph.get_related_documents(ent))

        # Deduplicate while preserving order
        seen: set[str] = set()
        merged: list[str] = []
        for doc in vector_results + graph_doc_ids:
            if doc not in seen:
                seen.add(doc)
                merged.append(doc)

        return merged[:top_k], {
            "entities_extracted": entities,
            "graph_docs": list(dict.fromkeys(graph_doc_ids)),
            "traversal_hops": self._max_hops,
        }
