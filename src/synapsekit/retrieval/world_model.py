"""World-model RAG with temporal and causal graph retrieval."""

from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, Protocol

from .._compat import run_sync
from ..embeddings.backend import SynapsekitEmbeddings
from ..llm._factory import make_llm
from ..llm.base import BaseLLM
from ..memory.conversation import ConversationMemory
from ..observability.tracer import TokenTracer
from ..rag.pipeline import RAGConfig, RAGPipeline
from .retriever import Retriever
from .vectorstore import InMemoryVectorStore

WorldModelBackend = Literal["in_memory", "neo4j", "memgraph", "kuzu"]
QueryStrategy = Literal["graph_first", "vector_first", "hybrid"]

_ENTITY_PATTERN = re.compile(
    r"(?:@[A-Za-z][\w.-]*|[A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){0,4})"
)
_DATE_PATTERN = re.compile(r"\b(?:20\d{2}-\d{2}-\d{2}|20\d{2}|Q[1-4]\s+20\d{2})\b")
_RELATION_HINTS: tuple[tuple[str, str], ...] = (
    ("led to", "caused"),
    ("caused", "caused"),
    ("unblocked", "unblocked"),
    ("blocked", "blocked"),
    ("depends on", "depends_on"),
    ("deprecated", "deprecated"),
    ("acquired", "acquired"),
    ("purchased", "purchased"),
    ("released", "released"),
    ("worked on", "worked_on"),
    ("introduced", "introduced"),
    ("touches", "touches"),
    ("uses", "uses"),
)


@dataclass(frozen=True)
class ExtractionPolicy:
    """Controls which structures the world-model extractor should emit."""

    entities: list[str] = field(
        default_factory=lambda: ["person", "org", "product", "event", "location", "concept"]
    )
    relations: list[str] | Literal["open_schema"] = "open_schema"
    temporal: bool = True
    causal: bool = True
    min_confidence: float = 0.55


@dataclass(frozen=True)
class EntityMention:
    """A normalized entity mention extracted from a document."""

    name: str
    type: str = "entity"
    aliases: tuple[str, ...] = ()
    confidence: float = 1.0


@dataclass(frozen=True)
class RelationMention:
    """A directed relation extracted from a document."""

    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    valid_at: datetime | None = None
    valid_until: datetime | None = None
    causal: bool = False


@dataclass(frozen=True)
class EventMention:
    """A temporal event extracted from a document."""

    name: str
    participants: tuple[str, ...] = ()
    timestamp: datetime | None = None
    confidence: float = 1.0


@dataclass(frozen=True)
class ExtractionResult:
    """Structured world-model facts produced for one source document."""

    entities: list[EntityMention] = field(default_factory=list)
    relations: list[RelationMention] = field(default_factory=list)
    events: list[EventMention] = field(default_factory=list)


@dataclass
class WorldModelNode:
    """Canonical graph node with provenance and alias tracking."""

    id: str
    name: str
    type: str = "entity"
    aliases: set[str] = field(default_factory=set)
    confidence: float = 1.0
    provenance: set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class WorldModelEdge:
    """Directed relationship edge with temporal and causal metadata."""

    id: str
    subject_id: str
    predicate: str
    object_id: str
    confidence: float = 1.0
    valid_at: datetime | None = None
    valid_until: datetime | None = None
    causal: bool = False
    provenance: set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class GraphQueryResult:
    """Subgraph selected for a query."""

    nodes: list[WorldModelNode]
    edges: list[WorldModelEdge]
    documents: list[str]
    query_entities: list[str]


@dataclass(frozen=True)
class WorldModelQueryResult:
    """Result returned by ``WorldModelRAG.query``."""

    query: str
    strategy: QueryStrategy
    subgraph: GraphQueryResult
    embeddings: list[dict[str, Any]]
    contexts: list[str]
    answer: str


class WorldModelExtractor(Protocol):
    """Extractor interface for pluggable LLM or rules-based pipelines."""

    async def extract(self, text: str, policy: ExtractionPolicy) -> ExtractionResult:
        """Extract entities, relations, and events from text."""
        ...


class EntityResolver:
    """Conservative entity resolver based on aliases and normalized names."""

    def __init__(self) -> None:
        self._alias_to_id: dict[str, str] = {}

    def resolve(self, entity: EntityMention, nodes: dict[str, WorldModelNode]) -> str | None:
        """Return an existing canonical node ID when confidence is high."""
        keys = [self._normalize(entity.name), *(self._normalize(alias) for alias in entity.aliases)]
        for key in keys:
            node_id = self._alias_to_id.get(key)
            if node_id in nodes:
                return node_id
        return None

    def remember(self, node: WorldModelNode) -> None:
        """Index a canonical node name and aliases for future resolution."""
        self._alias_to_id[self._normalize(node.name)] = node.id
        for alias in node.aliases:
            self._alias_to_id[self._normalize(alias)] = node.id

    @staticmethod
    def _normalize(value: str) -> str:
        value = value.casefold().strip()
        value = re.sub(r"^@", "", value)
        value = re.sub(r"\b(inc|llc|ltd|corp|corporation|company)\b\.?", "", value)
        value = re.sub(r"[^a-z0-9]+", " ", value)
        return " ".join(value.split())


class HeuristicWorldModelExtractor:
    """Dependency-free extractor for offline demos and deterministic tests."""

    async def extract(self, text: str, policy: ExtractionPolicy) -> ExtractionResult:
        entities = self._extract_entities(text)
        relations = self._extract_relations(text, entities, policy)
        events = self._extract_events(text, entities, policy)
        return ExtractionResult(entities=entities, relations=relations, events=events)

    def _extract_entities(self, text: str) -> list[EntityMention]:
        seen: set[str] = set()
        entities: list[EntityMention] = []
        for match in _ENTITY_PATTERN.finditer(text):
            name = match.group(0).strip(" .,;:()[]")
            if not self._valid_entity(name):
                continue
            key = name.casefold()
            if key in seen:
                continue
            seen.add(key)
            entities.append(EntityMention(name=name, type=self._guess_type(name), confidence=0.72))
        return entities

    def _extract_relations(
        self,
        text: str,
        entities: list[EntityMention],
        policy: ExtractionPolicy,
    ) -> list[RelationMention]:
        if len(entities) < 2:
            return []

        dates = self._extract_dates(text) if policy.temporal else []
        default_date = dates[0] if dates else None
        relations: list[RelationMention] = []
        entity_names = [entity.name for entity in entities]

        for sentence in self._sentences(text):
            sentence_entities = [name for name in entity_names if name in sentence]
            if len(sentence_entities) < 2:
                continue
            predicate = self._predicate_for(sentence)
            causal = predicate in {"caused", "unblocked"} or "led to" in sentence.casefold()
            if causal and not policy.causal:
                continue
            for left, right in pairwise(sentence_entities):
                relations.append(
                    RelationMention(
                        subject=left,
                        predicate=predicate,
                        object=right,
                        confidence=0.7,
                        valid_at=default_date,
                        causal=causal,
                    )
                )

        return relations

    def _extract_events(
        self,
        text: str,
        entities: list[EntityMention],
        policy: ExtractionPolicy,
    ) -> list[EventMention]:
        if not policy.temporal:
            return []
        dates = self._extract_dates(text)
        if not dates:
            return []
        participants = tuple(entity.name for entity in entities[:8])
        events: list[EventMention] = []
        for index, timestamp in enumerate(dates, start=1):
            events.append(
                EventMention(
                    name=f"event_{timestamp.date().isoformat()}_{index}",
                    participants=participants,
                    timestamp=timestamp,
                    confidence=0.65,
                )
            )
        return events

    @staticmethod
    def _valid_entity(name: str) -> bool:
        if len(name) < 2:
            return False
        blocked = {
            "A",
            "An",
            "And",
            "Between",
            "In",
            "It",
            "Q1",
            "Q2",
            "Q3",
            "Q4",
            "The",
            "This",
            "What",
            "Which",
            "Who",
        }
        return name not in blocked and not name.isdigit()

    @staticmethod
    def _guess_type(name: str) -> str:
        lowered = name.casefold()
        if lowered.startswith("@"):
            return "person"
        if any(token in lowered for token in (" inc", " corp", " llc", " gmbh", " ltd", " api")):
            return "org" if " api" not in lowered else "product"
        if any(token in lowered for token in ("release", "launch", "incident", "migration")):
            return "event"
        return "entity"

    @staticmethod
    def _predicate_for(sentence: str) -> str:
        lowered = sentence.casefold()
        for hint, predicate in _RELATION_HINTS:
            if hint in lowered:
                return predicate
        return "related_to"

    @staticmethod
    def _extract_dates(text: str) -> list[datetime]:
        dates: list[datetime] = []
        for match in _DATE_PATTERN.finditer(text):
            raw = match.group(0)
            if raw.startswith("Q"):
                quarter, year = raw.split()
                month = (int(quarter[1]) - 1) * 3 + 1
                dates.append(datetime(int(year), month, 1, tzinfo=UTC))
            elif len(raw) == 4:
                dates.append(datetime(int(raw), 1, 1, tzinfo=UTC))
            else:
                parsed = datetime.fromisoformat(raw)
                dates.append(parsed.replace(tzinfo=parsed.tzinfo or UTC))
        return dates

    @staticmethod
    def _sentences(text: str) -> list[str]:
        return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


class LLMWorldModelExtractor:
    """LLM-driven extractor that asks for strict JSON and validates it defensively."""

    _PROMPT = """\
Extract a world model from the text.
Return STRICT JSON with keys "entities", "relations", and "events".

Entity object: {{"name": str, "type": str, "aliases": [str], "confidence": float}}
Relation object: {{"subject": str, "predicate": str, "object": str, "confidence": float, "valid_at": ISO date|null, "valid_until": ISO date|null, "causal": bool}}
Event object: {{"name": str, "participants": [str], "timestamp": ISO date|null, "confidence": float}}

Allowed entity types: {entity_types}
Allowed relations: {relations}
Temporal extraction enabled: {temporal}
Causal extraction enabled: {causal}

Text:
{text}
"""

    def __init__(self, llm: BaseLLM) -> None:
        self._llm = llm

    async def extract(self, text: str, policy: ExtractionPolicy) -> ExtractionResult:
        prompt = self._PROMPT.format(
            entity_types=", ".join(policy.entities),
            relations=policy.relations,
            temporal=policy.temporal,
            causal=policy.causal,
            text=text,
        )
        response = await self._llm.generate(prompt)
        payload = self._loads(response)
        return ExtractionResult(
            entities=self._entities(payload.get("entities", []), policy),
            relations=self._relations(payload.get("relations", []), policy),
            events=self._events(payload.get("events", []), policy),
        )

    @staticmethod
    def _loads(response: str) -> dict[str, Any]:
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        try:
            payload = json.loads(cleaned.strip())
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _entities(rows: Any, policy: ExtractionPolicy) -> list[EntityMention]:
        entities: list[EntityMention] = []
        if not isinstance(rows, list):
            return entities
        for row in rows:
            if not isinstance(row, dict) or not row.get("name"):
                continue
            confidence = float(row.get("confidence", 1.0))
            if confidence < policy.min_confidence:
                continue
            aliases = row.get("aliases", [])
            entities.append(
                EntityMention(
                    name=str(row["name"]),
                    type=str(row.get("type", "entity")),
                    aliases=tuple(str(alias) for alias in aliases if isinstance(alias, str)),
                    confidence=confidence,
                )
            )
        return entities

    @staticmethod
    def _relations(rows: Any, policy: ExtractionPolicy) -> list[RelationMention]:
        relations: list[RelationMention] = []
        if not isinstance(rows, list):
            return relations
        for row in rows:
            if not isinstance(row, dict):
                continue
            subject = row.get("subject")
            predicate = row.get("predicate")
            obj = row.get("object")
            if not subject or not predicate or not obj:
                continue
            confidence = float(row.get("confidence", 1.0))
            if confidence < policy.min_confidence:
                continue
            causal = bool(row.get("causal", False))
            if causal and not policy.causal:
                continue
            relations.append(
                RelationMention(
                    subject=str(subject),
                    predicate=str(predicate),
                    object=str(obj),
                    confidence=confidence,
                    valid_at=_parse_datetime(row.get("valid_at")) if policy.temporal else None,
                    valid_until=_parse_datetime(row.get("valid_until"))
                    if policy.temporal
                    else None,
                    causal=causal,
                )
            )
        return relations

    @staticmethod
    def _events(rows: Any, policy: ExtractionPolicy) -> list[EventMention]:
        if not policy.temporal or not isinstance(rows, list):
            return []
        events: list[EventMention] = []
        for row in rows:
            if not isinstance(row, dict) or not row.get("name"):
                continue
            confidence = float(row.get("confidence", 1.0))
            if confidence < policy.min_confidence:
                continue
            participants = row.get("participants", [])
            events.append(
                EventMention(
                    name=str(row["name"]),
                    participants=tuple(
                        str(participant)
                        for participant in participants
                        if isinstance(participant, str)
                    ),
                    timestamp=_parse_datetime(row.get("timestamp")),
                    confidence=confidence,
                )
            )
        return events


class GraphBackend(Protocol):
    """Storage contract for world-model graph backends."""

    def upsert_entity(self, entity: EntityMention, doc_id: str) -> WorldModelNode:
        """Insert or merge an entity."""
        ...

    def upsert_relation(self, relation: RelationMention, doc_id: str) -> WorldModelEdge | None:
        """Insert or merge a relation."""
        ...

    def add_event(self, event: EventMention, doc_id: str) -> WorldModelNode:
        """Insert an event node and participant edges."""
        ...

    def query_subgraph(
        self,
        query: str,
        *,
        max_hops: int = 2,
        as_of: str | datetime | None = None,
        min_confidence: float = 0.0,
    ) -> GraphQueryResult:
        """Return a traversed subgraph relevant to a query."""
        ...

    def to_mermaid(self, result: GraphQueryResult | None = None) -> str:
        """Render the graph or subgraph as Mermaid."""
        ...


class InMemoryWorldGraphBackend:
    """In-memory bitemporal graph backend for world-model RAG."""

    def __init__(self, resolver: EntityResolver | None = None) -> None:
        self._resolver = resolver or EntityResolver()
        self.nodes: dict[str, WorldModelNode] = {}
        self.edges: dict[str, WorldModelEdge] = {}
        self._name_to_id: dict[str, str] = {}
        self._documents_by_node: dict[str, set[str]] = defaultdict(set)
        self._adjacency: dict[str, set[str]] = defaultdict(set)

    def upsert_entity(self, entity: EntityMention, doc_id: str) -> WorldModelNode:
        node_id = self._resolver.resolve(entity, self.nodes)
        now = datetime.now(UTC)
        if node_id is None:
            node_id = _slug(entity.name)
            node = WorldModelNode(
                id=node_id,
                name=entity.name,
                type=entity.type,
                aliases=set(entity.aliases),
                confidence=entity.confidence,
                provenance={doc_id},
                created_at=now,
                updated_at=now,
            )
            self.nodes[node_id] = node
        else:
            node = self.nodes[node_id]
            node.aliases.update(entity.aliases)
            node.confidence = max(node.confidence, entity.confidence)
            node.provenance.add(doc_id)
            node.updated_at = now

        self._name_to_id[self._normalize(entity.name)] = node.id
        for alias in entity.aliases:
            self._name_to_id[self._normalize(alias)] = node.id
        self._documents_by_node[node.id].add(doc_id)
        self._resolver.remember(node)
        return node

    def upsert_relation(self, relation: RelationMention, doc_id: str) -> WorldModelEdge | None:
        subject_id = self._resolve_name(relation.subject)
        object_id = self._resolve_name(relation.object)
        if subject_id is None or object_id is None:
            return None

        edge_id = f"{subject_id}:{_slug(relation.predicate)}:{object_id}"
        now = datetime.now(UTC)
        if edge_id not in self.edges:
            self.edges[edge_id] = WorldModelEdge(
                id=edge_id,
                subject_id=subject_id,
                predicate=relation.predicate,
                object_id=object_id,
                confidence=relation.confidence,
                valid_at=relation.valid_at,
                valid_until=relation.valid_until,
                causal=relation.causal,
                provenance={doc_id},
                created_at=now,
                updated_at=now,
            )
        else:
            edge = self.edges[edge_id]
            edge.confidence = max(edge.confidence, relation.confidence)
            edge.causal = edge.causal or relation.causal
            edge.valid_at = edge.valid_at or relation.valid_at
            edge.valid_until = edge.valid_until or relation.valid_until
            edge.provenance.add(doc_id)
            edge.updated_at = now

        self._adjacency[subject_id].add(object_id)
        self._adjacency[object_id].add(subject_id)
        self._documents_by_node[subject_id].add(doc_id)
        self._documents_by_node[object_id].add(doc_id)
        return self.edges[edge_id]

    def add_event(self, event: EventMention, doc_id: str) -> WorldModelNode:
        event_entity = EntityMention(
            name=event.name,
            type="event",
            confidence=event.confidence,
        )
        event_node = self.upsert_entity(event_entity, doc_id)
        for participant in event.participants:
            relation = RelationMention(
                subject=participant,
                predicate="participated_in",
                object=event.name,
                confidence=event.confidence,
                valid_at=event.timestamp,
            )
            self.upsert_relation(relation, doc_id)
        return event_node

    def query_subgraph(
        self,
        query: str,
        *,
        max_hops: int = 2,
        as_of: str | datetime | None = None,
        min_confidence: float = 0.0,
    ) -> GraphQueryResult:
        timestamp = _parse_datetime(as_of)
        seeds = self._query_seeds(query)
        if not seeds:
            seeds = self._fallback_seeds(query)

        visited: set[str] = set()
        selected_edges: dict[str, WorldModelEdge] = {}
        documents: set[str] = set()
        queue: deque[tuple[str, int]] = deque((seed, 0) for seed in seeds)

        while queue:
            node_id, depth = queue.popleft()
            if node_id in visited or node_id not in self.nodes:
                continue
            visited.add(node_id)
            documents.update(self._documents_by_node.get(node_id, set()))
            if depth >= max_hops:
                continue

            for edge in self._incident_edges(node_id):
                if edge.confidence < min_confidence or not _edge_active(edge, timestamp):
                    continue
                selected_edges[edge.id] = edge
                other = edge.object_id if edge.subject_id == node_id else edge.subject_id
                if other not in visited:
                    queue.append((other, depth + 1))

        nodes = [self.nodes[node_id] for node_id in sorted(visited)]
        edges = [selected_edges[edge_id] for edge_id in sorted(selected_edges)]
        return GraphQueryResult(
            nodes=nodes,
            edges=edges,
            documents=sorted(documents),
            query_entities=[self.nodes[seed].name for seed in seeds if seed in self.nodes],
        )

    def to_mermaid(self, result: GraphQueryResult | None = None) -> str:
        nodes = {node.id: node for node in (result.nodes if result else self.nodes.values())}
        edges = result.edges if result else list(self.edges.values())
        lines = ["graph LR"]
        for node in nodes.values():
            lines.append(f'    {node.id}["{_escape_mermaid(node.name)}"]')
        for edge in edges:
            if edge.subject_id not in nodes or edge.object_id not in nodes:
                continue
            label = edge.predicate
            if edge.causal:
                label = f"{label} (causal)"
            lines.append(f"    {edge.subject_id} -->|{_escape_mermaid(label)}| {edge.object_id}")
        return "\n".join(lines)

    def _query_seeds(self, query: str) -> list[str]:
        normalized_query = self._normalize(query)
        seeds: list[str] = []
        for name_key, node_id in self._name_to_id.items():
            if name_key and name_key in normalized_query:
                seeds.append(node_id)
        return list(dict.fromkeys(seeds))

    def _fallback_seeds(self, query: str) -> list[str]:
        terms = set(self._normalize(query).split())
        scored: list[tuple[int, str]] = []
        for node in self.nodes.values():
            score = len(terms & set(self._normalize(node.name).split()))
            if score:
                scored.append((score, node.id))
        return [node_id for _, node_id in sorted(scored, reverse=True)[:5]]

    def _resolve_name(self, name: str) -> str | None:
        return self._name_to_id.get(self._normalize(name))

    def _incident_edges(self, node_id: str) -> list[WorldModelEdge]:
        return [
            edge
            for edge in self.edges.values()
            if edge.subject_id == node_id or edge.object_id == node_id
        ]

    @staticmethod
    def _normalize(value: str) -> str:
        return EntityResolver._normalize(value)


class ExternalWorldGraphBackend:
    """Placeholder adapter for graph services not available in core installs."""

    def __init__(self, backend: WorldModelBackend, uri: str | None = None) -> None:
        self.backend = backend
        self.uri = uri

    def _unsupported(self) -> RuntimeError:
        return RuntimeError(
            f"{self.backend!r} world-model backend is not available in the core package yet. "
            "Use graph_backend='in_memory' or provide a custom GraphBackend implementation."
        )

    def upsert_entity(self, entity: EntityMention, doc_id: str) -> WorldModelNode:
        raise self._unsupported()

    def upsert_relation(self, relation: RelationMention, doc_id: str) -> WorldModelEdge | None:
        raise self._unsupported()

    def add_event(self, event: EventMention, doc_id: str) -> WorldModelNode:
        raise self._unsupported()

    def query_subgraph(
        self,
        query: str,
        *,
        max_hops: int = 2,
        as_of: str | datetime | None = None,
        min_confidence: float = 0.0,
    ) -> GraphQueryResult:
        raise self._unsupported()

    def to_mermaid(self, result: GraphQueryResult | None = None) -> str:
        raise self._unsupported()


class KuzuWorldGraphBackend(InMemoryWorldGraphBackend):
    """Optional Kuzu-backed world graph with an in-memory query mirror.

    Kuzu is intentionally optional. The in-memory mirror keeps the full
    ``GraphBackend`` behavior available in-process while entity/relation/event
    writes are mirrored to Kuzu when the package is installed.
    """

    def __init__(
        self,
        path: str | Path,
        resolver: EntityResolver | None = None,
    ) -> None:
        super().__init__(resolver=resolver)
        try:
            import kuzu
        except ImportError:
            raise ImportError("kuzu required: pip install synapsekit[kuzu]") from None

        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = kuzu.Database(str(self.path))
        self._conn = kuzu.Connection(self._db)
        self._ensure_kuzu_schema()

    def upsert_entity(self, entity: EntityMention, doc_id: str) -> WorldModelNode:
        node = super().upsert_entity(entity, doc_id)
        self._persist_entity(node)
        self._persist_document(doc_id)
        self._execute(
            "MATCH (d:Document {id: $doc_id}), (e:Entity {id: $entity_id}) "
            "MERGE (d)-[:MENTIONS]->(e)",
            {"doc_id": doc_id, "entity_id": node.id},
        )
        return node

    def upsert_relation(self, relation: RelationMention, doc_id: str) -> WorldModelEdge | None:
        edge = super().upsert_relation(relation, doc_id)
        if edge is None:
            return None
        self._persist_document(doc_id)
        self._persist_edge(edge)
        return edge

    def add_event(self, event: EventMention, doc_id: str) -> WorldModelNode:
        node = super().add_event(event, doc_id)
        self._persist_entity(node)
        self._persist_document(doc_id)
        return node

    def _ensure_kuzu_schema(self) -> None:
        statements = [
            """
            CREATE NODE TABLE IF NOT EXISTS Entity(
                id STRING,
                name STRING,
                kind STRING,
                aliases STRING,
                confidence DOUBLE,
                provenance STRING,
                created_at STRING,
                updated_at STRING,
                PRIMARY KEY (id)
            )
            """,
            "CREATE NODE TABLE IF NOT EXISTS Document(id STRING, PRIMARY KEY (id))",
            "CREATE REL TABLE IF NOT EXISTS MENTIONS(FROM Document TO Entity)",
            """
            CREATE REL TABLE IF NOT EXISTS RELATES(
                FROM Entity TO Entity,
                id STRING,
                predicate STRING,
                confidence DOUBLE,
                causal BOOL,
                provenance STRING,
                valid_at STRING,
                valid_until STRING
            )
            """,
        ]
        for statement in statements:
            self._execute(statement)

    def _persist_document(self, doc_id: str) -> None:
        self._execute("MERGE (:Document {id: $id})", {"id": doc_id})

    def _persist_entity(self, node: WorldModelNode) -> None:
        self._execute(
            """
            MERGE (e:Entity {id: $id})
            SET e.name = $name,
                e.kind = $kind,
                e.aliases = $aliases,
                e.confidence = $confidence,
                e.provenance = $provenance,
                e.created_at = $created_at,
                e.updated_at = $updated_at
            """,
            {
                "id": node.id,
                "name": node.name,
                "kind": node.type,
                "aliases": json.dumps(sorted(node.aliases)),
                "confidence": node.confidence,
                "provenance": json.dumps(sorted(node.provenance)),
                "created_at": node.created_at.isoformat(),
                "updated_at": node.updated_at.isoformat(),
            },
        )

    def _persist_edge(self, edge: WorldModelEdge) -> None:
        self._execute(
            """
            MATCH (s:Entity {id: $subject_id}), (o:Entity {id: $object_id})
            MERGE (s)-[r:RELATES {id: $id}]->(o)
            SET r.predicate = $predicate,
                r.confidence = $confidence,
                r.causal = $causal,
                r.provenance = $provenance,
                r.valid_at = $valid_at,
                r.valid_until = $valid_until
            """,
            {
                "id": edge.id,
                "subject_id": edge.subject_id,
                "object_id": edge.object_id,
                "predicate": edge.predicate,
                "confidence": edge.confidence,
                "causal": edge.causal,
                "provenance": json.dumps(sorted(edge.provenance)),
                "valid_at": edge.valid_at.isoformat() if edge.valid_at else None,
                "valid_until": edge.valid_until.isoformat() if edge.valid_until else None,
            },
        )

    def _execute(self, query: str, parameters: dict[str, Any] | None = None) -> Any:
        if parameters is None:
            return self._conn.execute(query)
        return self._conn.execute(query, parameters)


class Neo4jWorldGraphBackend(InMemoryWorldGraphBackend):
    """Optional Neo4j/Memgraph-backed world graph with an in-memory query mirror.

    Both Neo4j and Memgraph speak the Bolt protocol, so this one class serves
    both. Same shape as ``KuzuWorldGraphBackend``: writes are mirrored to the
    external store while reads (``query_subgraph``, ``to_mermaid``) are served
    from the in-memory graph inherited from the parent class.
    """

    def __init__(
        self,
        uri: str,
        *,
        username: str = "neo4j",
        password: str = "password",
        database: str | None = None,
        resolver: EntityResolver | None = None,
    ) -> None:
        super().__init__(resolver=resolver)
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise ImportError(
                "Neo4jWorldGraphBackend requires the 'neo4j' package. "
                "Install SynapseKit with the graph extra."
            ) from exc

        self._driver = GraphDatabase.driver(uri, auth=(username, password))
        self._database = database

    def upsert_entity(self, entity: EntityMention, doc_id: str) -> WorldModelNode:
        node = super().upsert_entity(entity, doc_id)
        self._persist_entity(node)
        self._persist_document(doc_id)
        self._run(
            "MATCH (d:WorldModelDocument {id: $doc_id}), (e:WorldModelEntity {id: $entity_id}) "
            "MERGE (d)-[:MENTIONS]->(e)",
            doc_id=doc_id,
            entity_id=node.id,
        )
        return node

    def upsert_relation(self, relation: RelationMention, doc_id: str) -> WorldModelEdge | None:
        edge = super().upsert_relation(relation, doc_id)
        if edge is None:
            return None
        self._persist_document(doc_id)
        self._persist_edge(edge)
        return edge

    def add_event(self, event: EventMention, doc_id: str) -> WorldModelNode:
        node = super().add_event(event, doc_id)
        self._persist_entity(node)
        self._persist_document(doc_id)
        return node

    def close(self) -> None:
        self._driver.close()

    def _persist_document(self, doc_id: str) -> None:
        self._run("MERGE (:WorldModelDocument {id: $id})", id=doc_id)

    def _persist_entity(self, node: WorldModelNode) -> None:
        self._run(
            """
            MERGE (e:WorldModelEntity {id: $id})
            SET e.name = $name,
                e.kind = $kind,
                e.aliases = $aliases,
                e.confidence = $confidence,
                e.provenance = $provenance,
                e.created_at = $created_at,
                e.updated_at = $updated_at
            """,
            id=node.id,
            name=node.name,
            kind=node.type,
            aliases=json.dumps(sorted(node.aliases)),
            confidence=node.confidence,
            provenance=json.dumps(sorted(node.provenance)),
            created_at=node.created_at.isoformat(),
            updated_at=node.updated_at.isoformat(),
        )

    def _persist_edge(self, edge: WorldModelEdge) -> None:
        self._run(
            """
            MATCH (s:WorldModelEntity {id: $subject_id}), (o:WorldModelEntity {id: $object_id})
            MERGE (s)-[r:WORLD_RELATES {id: $id}]->(o)
            SET r.predicate = $predicate,
                r.confidence = $confidence,
                r.causal = $causal,
                r.provenance = $provenance,
                r.valid_at = $valid_at,
                r.valid_until = $valid_until
            """,
            id=edge.id,
            subject_id=edge.subject_id,
            object_id=edge.object_id,
            predicate=edge.predicate,
            confidence=edge.confidence,
            causal=edge.causal,
            provenance=json.dumps(sorted(edge.provenance)),
            valid_at=edge.valid_at.isoformat() if edge.valid_at else None,
            valid_until=edge.valid_until.isoformat() if edge.valid_until else None,
        )

    def _run(self, query: str, **parameters: Any) -> None:
        with self._driver.session(database=self._database) as session:
            session.run(query, **parameters)


class CausalLinker:
    """Scores candidate causal edges with optional NeuroSymbolicAgent-style verifier."""

    def __init__(self, verifier: Any | None = None, threshold: float = 0.6) -> None:
        self._verifier = verifier
        self.threshold = threshold

    async def score(self, relation: RelationMention, text: str) -> float:
        """Return a plausibility score for a causal relation."""
        if not relation.causal:
            return relation.confidence
        if self._verifier is None:
            return relation.confidence
        verify = getattr(self._verifier, "score", None) or getattr(self._verifier, "verify", None)
        if not callable(verify):
            return relation.confidence
        result = verify(
            {
                "subject": relation.subject,
                "predicate": relation.predicate,
                "object": relation.object,
                "text": text,
            }
        )
        if hasattr(result, "__await__"):
            result = await result
        if isinstance(result, int | float):
            return float(result)
        if isinstance(result, dict):
            score = result.get("score") or result.get("confidence")
            if isinstance(score, int | float):
                return float(score)
        return relation.confidence


class HybridWorldModelRetriever:
    """Retriever that fuses graph traversal and vector search."""

    def __init__(
        self,
        graph_backend: GraphBackend,
        vector_retriever: Retriever,
        *,
        strategy: QueryStrategy = "hybrid",
        max_hops: int = 2,
        min_confidence: float = 0.0,
    ) -> None:
        self.graph_backend = graph_backend
        self.vector_retriever = vector_retriever
        self.strategy = strategy
        self.max_hops = max_hops
        self.min_confidence = min_confidence

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
        *,
        strategy: QueryStrategy | None = None,
        as_of: str | datetime | None = None,
    ) -> list[str]:
        results = await self.retrieve_with_scores(
            query,
            top_k=top_k,
            metadata_filter=metadata_filter,
            strategy=strategy,
            as_of=as_of,
        )
        return [str(result["text"]) for result in results]

    async def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
        *,
        strategy: QueryStrategy | None = None,
        as_of: str | datetime | None = None,
    ) -> list[dict[str, Any]]:
        active_strategy = strategy or self.strategy
        graph = self.graph_backend.query_subgraph(
            query,
            max_hops=self.max_hops,
            as_of=as_of,
            min_confidence=self.min_confidence,
        )
        vector_results = await self.vector_retriever.retrieve_with_scores(
            query,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )
        graph_results = self._graph_results(graph)
        fused = self._fuse(active_strategy, graph_results, vector_results, top_k)
        return fused

    @staticmethod
    def _graph_results(graph: GraphQueryResult) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for rank, doc_id in enumerate(graph.documents, start=1):
            results.append(
                {
                    "text": doc_id,
                    "score": 1.0 / rank,
                    "metadata": {
                        "source": doc_id,
                        "retrieval_source": "world_model_graph",
                        "query_entities": graph.query_entities,
                    },
                }
            )
        return results

    @staticmethod
    def _fuse(
        strategy: QueryStrategy,
        graph_results: list[dict[str, Any]],
        vector_results: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if strategy == "graph_first":
            ordered = [*graph_results, *vector_results]
        elif strategy == "vector_first":
            ordered = [*vector_results, *graph_results]
        else:
            ordered = _reciprocal_rank_fusion(graph_results, vector_results)

        seen: set[str] = set()
        merged: list[dict[str, Any]] = []
        for item in ordered:
            text = str(item.get("text", ""))
            if not text or text in seen:
                continue
            seen.add(text)
            merged.append(item)
            if len(merged) >= top_k:
                break
        return merged


class WorldModelRAG:
    """Facade for live causal/temporal graph + vector RAG."""

    def __init__(
        self,
        *,
        extraction: ExtractionPolicy | None = None,
        graph_backend: WorldModelBackend | GraphBackend = "in_memory",
        vector_store: Any | None = None,
        extractor: WorldModelExtractor | None = None,
        entity_resolver: EntityResolver | None = None,
        causal_linker: CausalLinker | None = None,
        llm: BaseLLM | None = None,
        model: str = "gpt-4o-mini",
        api_key: str = "",
        provider: str | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        retrieval_top_k: int = 5,
        max_hops: int = 2,
        strategy: QueryStrategy = "hybrid",
        system_prompt: str = "Answer using the graph subgraph and retrieved context. Cite source metadata when available.",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        trace: bool = True,
    ) -> None:
        self.extraction = extraction or ExtractionPolicy()
        self.llm = llm or make_llm(model, api_key, provider, system_prompt, temperature, max_tokens)
        self.graph_backend = self._make_graph_backend(graph_backend, entity_resolver)
        self.extractor = extractor or HeuristicWorldModelExtractor()
        self.causal_linker = causal_linker or CausalLinker()
        self.max_hops = max_hops
        self.strategy = strategy
        self._doc_counter = 0

        embeddings = SynapsekitEmbeddings(model=embedding_model)
        self.vector_store = (
            vector_store if vector_store is not None else InMemoryVectorStore(embeddings)
        )
        self.vector_retriever = Retriever(self.vector_store)
        self.retriever = HybridWorldModelRetriever(
            self.graph_backend,
            self.vector_retriever,
            strategy=strategy,
            max_hops=max_hops,
            min_confidence=self.extraction.min_confidence,
        )
        self._pipeline = RAGPipeline(
            RAGConfig(
                llm=self.llm,
                retriever=self.retriever,  # type: ignore[arg-type]
                memory=ConversationMemory(),
                tracer=TokenTracer(model=model, enabled=trace),
                retrieval_top_k=retrieval_top_k,
                system_prompt=system_prompt,
            )
        )

    @classmethod
    def kuzu(
        cls, path: str | Path, resolver: EntityResolver | None = None
    ) -> KuzuWorldGraphBackend:
        """Create an optional Kuzu graph backend for ``WorldModelRAG``."""

        return KuzuWorldGraphBackend(path, resolver=resolver)

    @classmethod
    def neo4j(
        cls,
        uri: str,
        *,
        username: str = "neo4j",
        password: str = "password",
        database: str | None = None,
        resolver: EntityResolver | None = None,
    ) -> Neo4jWorldGraphBackend:
        """Create an optional Neo4j/Memgraph graph backend for ``WorldModelRAG``."""

        return Neo4jWorldGraphBackend(
            uri, username=username, password=password, database=database, resolver=resolver
        )

    async def ingest(self, docs: list[str] | list[Any]) -> None:
        """Ingest documents into both the vector index and world graph."""
        # Collect all (text, metadata, doc_id) first, then add to the vector index
        # in a single batched call (embedding backends embed batches far more
        # efficiently than one text at a time).
        prepared: list[tuple[str, str]] = []
        batch_texts: list[str] = []
        batch_metadata: list[dict] = []
        for doc in docs:
            text, metadata = self._coerce_document(doc)
            if not text.strip():
                continue
            doc_id = self._doc_id(metadata)
            prepared.append((text, doc_id))
            batch_texts.append(text)
            batch_metadata.append({**metadata, "source": doc_id, "world_model_doc_id": doc_id})

        if batch_texts:
            await self.vector_retriever.add(batch_texts, batch_metadata)

        for text, doc_id in prepared:
            extraction = await self.extractor.extract(text, self.extraction)
            for entity in extraction.entities:
                self.graph_backend.upsert_entity(entity, doc_id)
            for event in extraction.events:
                self.graph_backend.add_event(event, doc_id)
            for relation in extraction.relations:
                score = await self.causal_linker.score(relation, text)
                if score >= self.extraction.min_confidence:
                    self.graph_backend.upsert_relation(
                        RelationMention(
                            subject=relation.subject,
                            predicate=relation.predicate,
                            object=relation.object,
                            confidence=score,
                            valid_at=relation.valid_at,
                            valid_until=relation.valid_until,
                            causal=relation.causal,
                        ),
                        doc_id,
                    )

    def ingest_sync(self, docs: list[str] | list[Any]) -> None:
        """Sync wrapper for ``ingest``."""
        run_sync(self.ingest(docs))

    def delete_by_metadata(self, key: str, values: set[str]) -> int:
        """Delete vector-store entries whose ``metadata[key]`` is in ``values``.

        Returns the number of deleted entries, or ``0`` if the underlying
        vector store does not support deletion.
        """
        delete = getattr(self.vector_store, "delete_by_metadata", None)
        if not callable(delete):
            return 0
        try:
            return int(delete(key, values))
        except NotImplementedError:
            return 0

    async def query(
        self,
        query: str,
        *,
        strategy: QueryStrategy | None = None,
        top_k: int | None = None,
        as_of: str | datetime | None = None,
    ) -> WorldModelQueryResult:
        """Run hybrid world-model retrieval and synthesize an answer."""
        active_strategy = strategy or self.strategy
        k = top_k if top_k is not None else self._pipeline.config.retrieval_top_k
        subgraph = self.graph_backend.query_subgraph(
            query,
            max_hops=self.max_hops,
            as_of=as_of,
            min_confidence=self.extraction.min_confidence,
        )
        embeddings = await self.retriever.retrieve_with_scores(
            query,
            top_k=k,
            strategy=active_strategy,
            as_of=as_of,
        )
        graph_context = self._format_graph_context(subgraph)
        vector_context = [str(item["text"]) for item in embeddings]
        prompt = self._query_prompt(query, graph_context, vector_context)
        answer = await self.llm.generate(prompt)
        return WorldModelQueryResult(
            query=query,
            strategy=active_strategy,
            subgraph=subgraph,
            embeddings=embeddings,
            contexts=[graph_context, *vector_context] if graph_context else vector_context,
            answer=answer,
        )

    def query_sync(
        self,
        query: str,
        *,
        strategy: QueryStrategy | None = None,
        top_k: int | None = None,
        as_of: str | datetime | None = None,
    ) -> WorldModelQueryResult:
        """Sync wrapper for ``query``."""
        return run_sync(self.query(query, strategy=strategy, top_k=top_k, as_of=as_of))

    def subgraph_to_mermaid(
        self,
        query: str | None = None,
        *,
        as_of: str | datetime | None = None,
    ) -> str:
        """Render the full graph or a query-selected subgraph as Mermaid."""
        if query is None:
            return self.graph_backend.to_mermaid()
        subgraph = self.graph_backend.query_subgraph(
            query,
            max_hops=self.max_hops,
            as_of=as_of,
            min_confidence=self.extraction.min_confidence,
        )
        return self.graph_backend.to_mermaid(subgraph)

    @staticmethod
    def _make_graph_backend(
        backend: WorldModelBackend | GraphBackend,
        resolver: EntityResolver | None,
    ) -> GraphBackend:
        if isinstance(backend, str):
            if backend == "in_memory":
                return InMemoryWorldGraphBackend(resolver=resolver)
            if backend == "kuzu":
                return KuzuWorldGraphBackend(Path.home() / ".synapsekit" / "world_model.kuzu")
            if backend in ("neo4j", "memgraph"):
                return Neo4jWorldGraphBackend("bolt://localhost:7687", resolver=resolver)
            return ExternalWorldGraphBackend(backend)
        return backend

    def _doc_id(self, metadata: dict[str, Any]) -> str:
        source = metadata.get("source") or metadata.get("id")
        if source:
            return str(source)
        self._doc_counter += 1
        return f"doc_{self._doc_counter}"

    @staticmethod
    def _coerce_document(doc: Any) -> tuple[str, dict[str, Any]]:
        if isinstance(doc, str):
            return doc, {}
        text = getattr(doc, "text", None)
        metadata = getattr(doc, "metadata", None)
        if isinstance(text, str):
            return text, dict(metadata or {})
        if isinstance(doc, dict):
            return str(doc.get("text", "")), dict(doc.get("metadata", {}))
        return str(doc), {}

    @staticmethod
    def _format_graph_context(subgraph: GraphQueryResult) -> str:
        if not subgraph.nodes and not subgraph.edges:
            return ""
        node_names = ", ".join(node.name for node in subgraph.nodes)
        edge_lines = []
        node_by_id = {node.id: node for node in subgraph.nodes}
        for edge in subgraph.edges:
            subject = node_by_id.get(edge.subject_id)
            obj = node_by_id.get(edge.object_id)
            if subject and obj:
                marker = " causal" if edge.causal else ""
                edge_lines.append(f"- {subject.name} -[{edge.predicate}{marker}]-> {obj.name}")
        return "World-model subgraph:\nEntities: " + node_names + "\n" + "\n".join(edge_lines)

    @staticmethod
    def _query_prompt(query: str, graph_context: str, vector_context: list[str]) -> str:
        vector_block = "\n\n".join(f"<document>\n{text}\n</document>" for text in vector_context)
        return (
            "Answer the question using the world-model graph and retrieved documents.\n\n"
            f"{graph_context or 'World-model subgraph: none'}\n\n"
            f"Retrieved documents:\n{vector_block or 'No vector context.'}\n\n"
            f"Question: {query}"
        )


def _reciprocal_rank_fusion(
    graph_results: list[dict[str, Any]],
    vector_results: list[dict[str, Any]],
    *,
    constant: int = 60,
) -> list[dict[str, Any]]:
    scores: dict[str, float] = defaultdict(float)
    payloads: dict[str, dict[str, Any]] = {}
    for results in (graph_results, vector_results):
        for rank, item in enumerate(results, start=1):
            text = str(item.get("text", ""))
            if not text:
                continue
            scores[text] += 1.0 / (constant + rank)
            payloads.setdefault(text, dict(item))
    fused: list[dict[str, Any]] = []
    for text, score in sorted(scores.items(), key=lambda row: row[1], reverse=True):
        item = payloads[text]
        item["score"] = score
        metadata = dict(item.get("metadata") or {})
        metadata.setdefault("retrieval_source", "world_model_hybrid")
        item["metadata"] = metadata
        fused.append(item)
    return fused


def _parse_datetime(value: Any) -> datetime | None:
    if value is None or isinstance(value, datetime):
        if isinstance(value, datetime) and value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=parsed.tzinfo or UTC)


def _edge_active(edge: WorldModelEdge, as_of: datetime | None) -> bool:
    if as_of is None:
        return True
    if edge.valid_at is not None and edge.valid_at > as_of:
        return False
    return not (edge.valid_until is not None and edge.valid_until <= as_of)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.casefold()).strip("_")
    return f"wm_{slug or 'node'}"


def _escape_mermaid(value: str) -> str:
    return value.replace('"', "'").replace("|", "/")
