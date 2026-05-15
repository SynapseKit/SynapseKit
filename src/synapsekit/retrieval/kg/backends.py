"""Graph storage backends for Knowledge Graph multi-hop retrieval."""

from __future__ import annotations

from typing import Protocol


class BaseGraphStore(Protocol):
    """Protocol for Knowledge Graph storage backends."""

    def add_triple(self, subject: str, predicate: str, obj: str, confidence: float = 1.0) -> None:
        """Add a directed relationship triple to the graph."""
        ...

    def add_document_link(self, entity: str, doc_id: str) -> None:
        """Link an entity to a document ID."""
        ...

    def get_neighbors(
        self, entity: str, max_hops: int = 1, min_confidence: float = 0.0
    ) -> set[str]:
        """Return connected entities within max_hops using depth-first traversal."""
        ...

    def get_related_documents(self, entity: str) -> list[str]:
        """Return document IDs linked to the entity."""
        ...


class NetworkXStore:
    """In-memory Knowledge Graph store using NetworkX. Recommended for development."""

    def __init__(self) -> None:
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "networkx is required for NetworkXStore. "
                "Install it with `pip install networkx` or `pip install synapsekit[graph]`."
            ) from None

        self.graph = nx.DiGraph()
        self._entity_to_docs: dict[str, set[str]] = {}

    def add_triple(self, subject: str, predicate: str, obj: str, confidence: float = 1.0) -> None:
        """Add a directed relationship triple to the graph."""
        self.graph.add_node(subject)
        self.graph.add_node(obj)
        self.graph.add_edge(subject, obj, predicate=predicate, confidence=confidence)

    def add_document_link(self, entity: str, doc_id: str) -> None:
        """Link an entity to a document ID."""
        self._entity_to_docs.setdefault(entity, set()).add(doc_id)

    def get_neighbors(
        self, entity: str, max_hops: int = 1, min_confidence: float = 0.0
    ) -> set[str]:
        """Return connected entities within max_hops using depth-first traversal."""
        if entity not in self.graph:
            return set()

        visited: set[str] = set()

        def dfs(current_node: str, current_depth: int) -> None:
            if current_depth > max_hops:
                return
            visited.add(current_node)
            for neighbor in self.graph.successors(current_node):
                edge_data = self.graph.get_edge_data(current_node, neighbor)
                confidence = edge_data.get("confidence", 1.0)
                if confidence >= min_confidence and neighbor not in visited:
                    dfs(neighbor, current_depth + 1)
            # Also traverse undirected / backwards for related entities context if useful,
            # but strictly following directed logic. Let's do undirected neighbors for RAG contexts:
            for neighbor in self.graph.predecessors(current_node):
                edge_data = self.graph.get_edge_data(neighbor, current_node)
                confidence = edge_data.get("confidence", 1.0)
                if confidence >= min_confidence and neighbor not in visited:
                    dfs(neighbor, current_depth + 1)

        dfs(entity, 0)
        visited.discard(entity)
        return visited

    def get_related_documents(self, entity: str) -> list[str]:
        """Return document IDs linked to the entity."""
        return list(self._entity_to_docs.get(entity, set()))


class Neo4jStore:
    """Persistent Knowledge Graph store using Neo4j. Recommended for production."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError(
                "neo4j is required for Neo4jStore. "
                "Install it with `pip install neo4j` or `pip install synapsekit[graph]`."
            ) from None

        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self._driver.close()

    def add_triple(self, subject: str, predicate: str, obj: str, confidence: float = 1.0) -> None:
        """Add a directed relationship triple to the graph."""
        query = (
            "MERGE (s:Entity {name: $subject}) "
            "MERGE (o:Entity {name: $obj}) "
            "MERGE (s)-[r:RELATIONSHIP {type: $predicate}]->(o) "
            "ON CREATE SET r.confidence = $confidence "
            "ON MATCH SET r.confidence = CASE WHEN $confidence > r.confidence THEN $confidence ELSE r.confidence END"
        )
        with self._driver.session() as session:
            session.run(query, subject=subject, obj=obj, predicate=predicate, confidence=confidence)

    def add_document_link(self, entity: str, doc_id: str) -> None:
        """Link an entity to a document ID."""
        query = (
            "MERGE (e:Entity {name: $entity}) "
            "MERGE (d:Document {id: $doc_id}) "
            "MERGE (e)-[:MENTIONED_IN]->(d)"
        )
        with self._driver.session() as session:
            session.run(query, entity=entity, doc_id=doc_id)

    def get_neighbors(
        self, entity: str, max_hops: int = 1, min_confidence: float = 0.0
    ) -> set[str]:
        """Return connected entities within max_hops using graph traversal."""
        # Using Cypher variable-length path to find neighbors up to max_hops
        query = (
            f"MATCH (e:Entity {{name: $entity}})-[r:RELATIONSHIP*1..{max_hops}]-(neighbor:Entity) "
            "WHERE all(rel IN r WHERE rel.confidence >= $min_confidence) "
            "RETURN DISTINCT neighbor.name AS name"
        )

        with self._driver.session() as session:
            result = session.run(query, entity=entity, min_confidence=min_confidence)
            return {record["name"] for record in result}

    def get_related_documents(self, entity: str) -> list[str]:
        """Return document IDs linked to the entity."""
        query = (
            "MATCH (e:Entity {name: $entity})-[:MENTIONED_IN]->(d:Document) RETURN d.id AS doc_id"
        )
        with self._driver.session() as session:
            result = session.run(query, entity=entity)
            return [record["doc_id"] for record in result]
