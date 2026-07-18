# World-Model RAG

World-model RAG builds a live graph of entities, events, temporal facts, and causal
relationships while also indexing the same documents for vector search. It is useful
when a question needs structure as well as semantic similarity, such as "what caused
this release?" or "which services depend on a deprecated API?"

```python
from synapsekit import ExtractionPolicy, WorldModelRAG

wm = WorldModelRAG(
    extraction=ExtractionPolicy(
        entities=["person", "org", "product", "event", "location"],
        relations="open_schema",
        temporal=True,
        causal=True,
    ),
    graph_backend="in_memory",
    model="gpt-4o-mini",
    api_key="sk-...",
)

await wm.ingest(
    [
        "Alice worked on Search API in 2026. Search API led to the v1.5 release.",
        "Billing Service depends on Deprecated API.",
    ]
)

result = await wm.query(
    "What did Alice work on that led to v1.5?",
    strategy="graph_first",
    as_of="2026-12-01",
)

print(result.answer)
print(wm.subgraph_to_mermaid("Alice v1.5"))
```

## Components

- `WorldModelRAG` is the facade for ingestion, hybrid querying, and Mermaid export.
- `ExtractionPolicy` controls entity types, relation schema, temporal extraction, causal links, and minimum confidence.
- `HeuristicWorldModelExtractor` provides dependency-free extraction for local demos and tests.
- `LLMWorldModelExtractor` asks a `BaseLLM` for strict JSON entities, relations, and events.
- `EntityResolver` conservatively merges aliases such as `Alice Chen` and `@alice`.
- `InMemoryWorldGraphBackend` stores bitemporal nodes and edges with provenance.
- `HybridWorldModelRetriever` fuses graph traversal and vector results with reciprocal rank fusion.

## Backends

The core package ships with `graph_backend="in_memory"` so world-model RAG works
without external services. `graph_backend="kuzu"` persists to a local embedded Kuzu
database, and `graph_backend="neo4j"` / `"memgraph"` persist to a Neo4j or Memgraph
instance over Bolt (`pip install synapsekit[graph]`; connection defaults to
`bolt://localhost:7687`, overridable via the `NEO4J_URI`, `NEO4J_USERNAME`, and
`NEO4J_PASSWORD` environment variables, or by passing `WorldModelRAG.neo4j(uri, ...)`
directly as `graph_backend` for full control). All three persistent backends serve
reads from an in-memory mirror built up as documents are ingested. Any other backend
name raises a clear runtime error unless you provide a custom `GraphBackend`
implementation.

## Time Travel

Relations and events can carry `valid_at` and `valid_until`. Query with `as_of` to
exclude facts that were not valid at that point:

```python
march = await wm.query("What led to v1.5?", as_of="2026-03-01")
may = await wm.query("What led to v1.5?", as_of="2026-05-01")
```

The graph also tracks transaction time through `created_at` and `updated_at` on each
node and edge.
