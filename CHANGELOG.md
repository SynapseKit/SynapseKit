# Changelog

All notable changes to SynapseKit are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
SynapseKit uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Security

- **Bumped 8 dependencies off known-vulnerable versions** — a full OSV audit of the resolved dependency graph (all extras + all groups, 534 packages) flagged 8 packages carrying active advisories; all are now floored at their patched releases and the lockfile re-resolves to 0 known vulnerabilities. Direct-dependency floors raised: `cryptography` → `>=50.0.0` (PYSEC-2026-3552), `pillow` → `>=12.3.0`, `pyasn1` → `>=0.6.4`, `aiohttp` (`http` extra) → `>=3.14.3`, `gitpython` (`git` extra) → `>=3.1.57`, `mcp` (`mcp` extra) → `>=1.28.1`. Transitive floors: `httplib2` → `>=0.32.0` (added to the `gdrive`/`gsheets`/`gcal-tool` extras that pull `google-api-python-client`, and to the `all` extra) and `setuptools` → `>=83.0.0` (build-time); both are also pinned via `[tool.uv] constraint-dependencies` so CI and the Docker images can't regress. No SynapseKit public API changed.

### Added
- **SynapseKit Live — zero-dependency real-time run dashboard** (#896, phase 1–2) — watch what an agent/RAG/graph is doing, live in your browser, with no extra dependencies. A single in-process `EventBus` (`synapsekit.live`) that every finished observe span publishes to; a stdlib `http.server` + **Server-Sent Events** stream (no FastAPI/uvicorn/websockets); and one self-contained `dashboard.html` (light theme, SynapseKit green, Story/Trace toggle, live cost/token meters) served from the wheel. Enable with `SYNAPSEKIT_LIVE=1` (auto-starts on the first span), `synapsekit ui --live`, or `synapsekit.live.enable()`. Bound to `127.0.0.1` and token-gated — nothing leaves the machine. Publishing is a no-op when Live is disabled, so there is zero overhead in production. Demo: `python examples/live_dashboard.py`. **Auto-instruments the paths observe spans miss** (#898): tools (and MCP tools, via `BaseTool`), agent memory / DB reads & writes, knowledge graphs (`WorldModelRAG` + property-graph backends), and the knowledge mesh — by wrapping their base/concrete async methods (existing subclasses + future ones via `__init_subclass__`), so DB/graph/tool/MCP activity streams to the dashboard automatically, async contract preserved. Also covers **data loaders** (sync `load`/`aload` on the common concrete loaders) and **embeddings** (`embed`), so a full run — loader → embeddings → vector search → tools/MCP → memory/DB → knowledge graph → LLM — streams end to end. Demo: `python examples/live_all_features.py`. The dashboard adds **Activity/Logs/Errors tabs** with click-to-expand event detail (LLM prompt + response, tracebacks, attributes), a **subsystem activity strip**, cost/token meters, a **latency sparkline**, a **daily-budget gauge**, signed-audit + swarm events, and **human-in-the-loop approvals** — `synapsekit.live.request_approval("send_email", ...)` blocks until you click Approve/Deny in the browser (`examples/live_hitl.py`). Python `logging` is bridged into the feed. Run **Export** downloads the run as JSON. Adds a **flame graph** of nested spans (agent/graph/RAG traces) and a **knowledge/run graph canvas** (force-directed; auto-populated with **real entities/relations** from `WorldModelRAG`/property-graph ingest, plus the live run-transition graph, or fed manually via `synapsekit.live.publish_graph(nodes, edges)`). The layout is fit-to-screen — the feed and telemetry are independently scrollable and the page scrolls to the graphs below. Async-safe: concurrent operations all stream (thread-safe bus, unique sequence ids). **Run history** — `synapsekit.live.new_run(label)` demarcates runs and the dashboard's **Runs** dropdown browses/filters past runs. A paced, narrated **`examples/live_showcase.py`** walks every subsystem + every UI feature across labelled runs for demos/recordings.
- **Release-validation harness (`release_check`)** — a reusable, no-mock check that a release build actually works end to end, run via `make release-check` (offline) or the scheduled `release-validation` workflow (offline + live). Three layers, each reporting pass/skip/fail into one markdown/JSON report: **core-import** (`import synapsekit` + CLI in a fresh interpreter with every optional dependency blocked — the same guarantee the core Docker image enforces), **export-surface** (every `synapsekit.__all__` name resolves), and **functional** (the real `smoke_test.py`: loaders, splitters, graph, tools, embeddings, vector stores, plus live LLM completions in `--live` mode when API keys are present). Exits non-zero on any failure; live checks skip cleanly without keys.
- **Official Docker images on GitHub Container Registry** (#874) — `docker pull ghcr.io/synapsekit/synapsekit` and run SynapseKit with no local Python setup. Two variants from one multi-stage, uv-based, non-root `Dockerfile`: `:latest` / `:<version>` (core lib + CLI, multi-arch amd64+arm64) and `:all` / `:<version>-all` (all extras baked in). A matching image publishes automatically on every GitHub Release, and each image is smoke-tested (`synapsekit --version` + import) before it is pushed. The Python version is a build arg (`PYTHON_VERSION`, core images build on 3.13); SynapseKit is verified to import on Python 3.12/3.13/3.14 and the CI test suite runs across 3.11–3.13.
- **`okf_to_world_model` — OKF → WorldModel graph adapter** (#825) — ingest an Open Knowledge Format bundle as a *knowledge graph*, not just flat chunks. Consumes the `Document` list from `OpenKnowledgeFormatLoader` (whose `linked_concepts` already resolved the cross-links) and builds one `WorldModelNode` per concept + one `WorldModelEdge` per link, with **no** lossy LLM/heuristic extraction. Node ids are deterministic (`_slug(concept_path)`) and written straight into the backend, bypassing the entity resolver so OKF's explicit ids are never fuzzily merged; frontmatter (`resource`/`tags`/`timestamp`/`okf_type`) maps onto node metadata and `timestamp` seeds edge `valid_at`. Idempotent re-ingestion, works against `InMemoryWorldGraphBackend` and `KuzuWorldGraphBackend` (persists where supported). `WorldModelNode`/`WorldModelEdge` gain an additive `metadata` dict. New `KnowledgeMesh.ingest_okf(path)` ingests a bundle end-to-end (vector-index bodies + explicit graph) for `graph_first`/`hybrid` retrieval via `WorldModelRAG`. Completes the OKF story: point SynapseKit at any bundle → portable, cloud-neutral graph RAG.
- **`OpenKnowledgeFormatLoader` (alias `OKFLoader`)** (#824) — a loader for Google's vendor-neutral [Open Knowledge Format (OKF v0.1)](https://github.com/GoogleCloudPlatform/knowledge-catalog/tree/main/okf): point it at an OKF bundle (a directory of Markdown concept files with YAML frontmatter) and get one `Document` per concept, with the standard frontmatter (`type`, `title`, `description`, `resource`, `tags`, `timestamp`) lifted into metadata and Markdown cross-links resolved into `metadata["linked_concepts"]` (bundle-relative paths). Handles single-file or nested bundles, `index.md` navigation-stub skipping (`include_index=`), spec-conformant `type`-required filtering (`require_type=`), deterministic ordering, and malformed frontmatter (warns, never crashes). Async-first (`aload()` offloads to a thread). New `okf` extra (`pip install synapsekit[okf]`, PyYAML only). Reuses the mesh markdown frontmatter parsing — no duplicate parser. The resolved link structure feeds the #825 OKF→WorldModel graph adapter for extraction-free graph RAG.
- **`GroundedSignal` / `SignalSource` provenance primitive** (#822) — a small, stateless value type (`synapsekit.provenance`, exported at top level) that tags any learning-relevant number (reward, quality, cost) with whether it came from an independent source (`EXTERNAL_OVERRIDE`) or was self-reported by the agent being scored (`SELF_REPORTED`). A strict two-tier split with a derived, read-only `grounded` property — no middle tier. `AgentSwarm` now uses it end-to-end: every auction emits a **replayable receipt** (`swarm.trace[-1]`) carrying `task_id`, each bid's `reputation_prior` (the snapshot it was scored against, with a `version` and `grounded_fraction`), `selected_roles`/`rejected_roles`, `budget_allocated`/`budget_consumed`, the `outcome_score_source`, and the `learning_rule`. A winning agent's own `estimated_quality` can appear as evidence but can never mark reputation `grounded` on its own. `ReputationSnapshot` gains additive `version`/`grounded`/`grounded_fraction` fields, and `Reputation.record_outcome` gains an optional `quality_signal` parameter (float callers unaffected). Opt-in strict enforcement via `MarketPolicy(require_grounded_reward=True)` no-ops the reputation update for an ungrounded outcome; the default keeps updating as before. Grounding audits documented for `CostQualityRouter` (not applicable — inputs are externally measured) and `SelfImprovingAgent` (accept gate is grounded via `EvalSuite`; feedback-provenance tagging is a deferred follow-up). Non-breaking, verified against the full suite. Originated from a review discussion on #734 with @clementineCU.
- **`OllamaLLM` now supports a custom `host`** (#858) — a new optional `host` parameter lets it target a remote or non-default Ollama server instead of only `http://localhost:11434`. Covered by a real Ollama testcontainer that runs actual local inference.
- **`DynamoDBLoader` now supports DynamoDB-compatible endpoints** (#855) — a new optional `endpoint_url` parameter lets it load from DynamoDB Local and LocalStack, not just AWS DynamoDB. Covered by real DynamoDB Local testcontainers tests.
- **`S3Loader` now supports S3-compatible endpoints** (#851) — a new optional `endpoint_url` parameter (with automatic path-style addressing) lets it load from MinIO, Cloudflare R2, DigitalOcean Spaces, and LocalStack, not just AWS S3. Covered by real MinIO testcontainers tests.

- Spec test for a replayable `AgentSwarm` auction receipt (`tests/agents/test_agent_swarm_receipt.py`, `xfail` pending implementation) — captures task_id, reputation-prior version, budget consumed, and outcome-score provenance per auction so a reviewer can tell whether a market win reflects a real outcome or a stale/self-reported score; added as an acceptance criterion on #734 (h/t @clementineCU)
- **`WorldModelRAG` Neo4j/Memgraph backend** — `Neo4jWorldGraphBackend` (Bolt; serves both Neo4j and Memgraph) with write-through persistence and live bounded-hop Cypher reads; selectable via `graph_backend="neo4j"` / `"memgraph"` (honoring `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD`) or `WorldModelRAG.neo4j(...)`; plus a 10k-document offline demo and a hybrid-vs-vector-only retrieval accuracy benchmark with a CI regression gate; completes the remaining #735 acceptance criteria; contributed by [@Abhay-Mmmm](https://github.com/Abhay-Mmmm)
- **Signed portable agent marketplace** (#751) — a deterministic, Ed25519-signed `.agent` bundle format with per-file SHA-256 verification, a safe install flow (verifies from an immutable in-memory snapshot before atomic extraction; rejects path traversal, symlinks, Windows device names/ADS, duplicates, and case-insensitive collisions; enforces archive-size/file-count limits and requires an explicit sandbox to run), a `synapsekit agent` CLI (keygen/pack/verify/install/run), and a self-hostable file-backed registry with signed reviews and eval-based ranking (`synapsekit.marketplace`); contributed by [@DhruvGarg111](https://github.com/DhruvGarg111)
- **Universal Memory Protocol (UMP) reference implementation** (#742) — a provider-agnostic memory document standard: `UMPDocument`/`UMPReader`/`UMPWriter` (async, YAML-frontmatter with `[[wikilink]]` extraction), a `UMPValidator`, and format adapters for `CLAUDE.md`, Cursor, Aider, and Continue (`synapsekit.ump`); contributed by [@DhruvGarg111](https://github.com/DhruvGarg111)
- **Digital Twin Agent** (#745) — `DigitalTwinAgent` learns a versioned `StyleProfile` (tone/structure/vocabulary) and drafts commit messages, PR descriptions, and reviews in the user's voice; a `VoiceMatcher` scores candidate drafts (n-gram/vocabulary/structure, optional LLM judge) and an enforced `DelegationPolicy` gates auto-sending (`never_send_auto` is a hard block, `draft_with_approval` requires explicit approval) (`synapsekit.twin`); contributed by [@DhruvGarg111](https://github.com/DhruvGarg111)
- **Time-Travel Codebase** (#746) — `TimeTravelAgent` reasons across a repo's evolution: a `GitBackend` with `as_of(date)` scoping, an AST-based `EvolutionIndex` (symbol/file change graph, timeline, `#NNN` PR linkage), a `DriftDetector` for abstractions whose justification has drifted, and a `DiffNarrativeGenerator` for markdown evolution timelines; all git subprocess work is offloaded off the event loop (`synapsekit.timetravel`); contributed by [@DhruvGarg111](https://github.com/DhruvGarg111)

### Testing

- **Self-evolving acceptance gate** (#732) — `benchmarks/self_evolving_bench.py` drives one real `SelfImprovingAgent` through 5 offline observe→propose→validate→canary cycles against a synthetic behavioural task suite, and a new `self-evolving-gate` CI job hard-enforces the acceptance targets: held-out accuracy climbs from a genuinely-fallible baseline (**37.5%**, ≤50% required) to **100%** (≥90% required), a **+62.5pp** uplift (≥35pp required), monotonically non-decreasing across every cycle. The eval gate is held out (8 tasks the proposer never sees) and load-bearing: from cycle 2 the proposer also emits a deliberately-bad decoy that strips the learned directives, and every decoy is *blocked* with a recorded `block_reason` (4 blocked), proving the gate decides rather than rubber-stamps. Every accepted patch is reversible (`rollback` restores the exact pre-patch accuracy) and every patch signature survives a JSONL round-trip through `AgentEvolutionAuditLog`. Fully deterministic (seeded, no keys, no network); nothing under `src/` changes. Contributed by [@Satiankit96](https://github.com/Satiankit96).
- **Neuro-symbolic acceptance gate** (#733) — `benchmarks/neuro_symbolic_bench.py` compares LLM-only direct answers against the real `NeuroSymbolicAgent` + `SympyBackend` on 20 arithmetic problems: solver-grounded answers score **100%** vs **~60%** for the direct baseline (**40pp uplift**). A new `neuro-symbolic-gate` CI job enforces neuro-symbolic ≥95%, uplift ≥25pp, and a genuinely-fallible baseline. (Surfaced #911: `verify()` doesn't check the answer against the solver result.)
- **ComputerUseAgent end-to-end acceptance** (#737) — `examples/computer_use_form_to_slack.py` drives the real agent loop to fill a legacy invoice form and post the total to Slack (in-process fake screen + scripted action model, no browser/keys), and `tests/computer_use/test_session_replay.py` proves the recorded session is **replayable** (re-running the recorded actions reproduces the end state). With the 3 wired providers, safety-block tests, and the mandatory safety guide already shipped, this completes #737.
- **`AgentSwarm` market-routing acceptance gate** (#734) — a deterministic 100-task benchmark (`benchmarks/agent_swarm_bench.py`) with a CI gate (`agent-swarm-market-gate`) that hard-enforces the two quantitative targets that previously had no code against them: the market swarm beats a hardcoded round-robin `AgentFederation` on mean outcome quality by **≥15%** (measured ~+41%) and costs **≥25% less** than always-routing-to-the-best-model (~74% less), via emergent per-category specialization. Plus a behavioral test proving `AgentSwarm` drives the `PrometheusMetrics` dashboard signals end-to-end (bids/sec, win rate per agent, avg reward). Completes the remaining #734 acceptance criteria (the replayable receipt + grounding half shipped in #822).
- **Real testcontainers integration tests now cover every external backend, replacing `MagicMock` throughout** (#829) — vector stores (pgvector, redis, elasticsearch, qdrant, weaviate, milvus, cassandra, clickhouse, opensearch, mongodb), memory + graph checkpointers (postgres/redis), loaders (S3/MinIO, SQL, Mongo, Elasticsearch, Redis, DynamoDB Local, Azure Blob/Azurite), and LLMs (OpenAI + Anthropic via respx HTTP contract, Ollama via real local inference). Every test does real inserts + real queries + reconnect reads, runs in the CI `integration` job (heavy Ollama isolated), and asserts the async-first contract. The conversion surfaced and fixed **~39 real production bugs** — all previously hidden by mocks — tracked in the per-backend entries below. Deferred (documented): Supabase (needs full hosted PostgREST+JWT+Kong stack), Gemini (old SDK is gRPC-only → not respx-able), OpenAI-compatible clones (share the OpenAI code path), and llamacpp/gpt4all/vllm.

### Fixed

- **`requires-python` corrected to `>=3.11`** (#884) — 10 modules (`mesh`, `retrieval`, `timetravel`, …) use `from datetime import UTC`, which only exists on Python 3.11+, so `import synapsekit` actually raised `ImportError` on 3.10 despite the metadata claiming `>=3.10`. Bumped the floor (and dropped the 3.10 classifier) to match reality; the CI test matrix now runs 3.11–3.13. Caught by the new Python-version CI matrix.
- **`from synapsekit import SupabaseLoader` now works** — `SupabaseLoader` was listed in the top-level `__all__` but never added to the lazy-import map, so accessing it raised `AttributeError` even though the class exists in `synapsekit.loaders`. Wired it into `_LAZY_IMPORTS`, and added a `tests/test_public_api_surface.py` guard that asserts every name in `__all__` resolves (so a future export/lazy-map drift can't ship). Surfaced by the new release-validation import-surface check.
- **`import synapsekit` no longer requires the `httpx` or `PyYAML` extras** — several eagerly-imported modules did module-level imports of optional dependencies, so a bare `pip install synapsekit` (no extras) failed to import the whole package: `NotionTool` (`synapsekit.agents.tools`) imported `httpx`, and the UMP parser/adapters (`synapsekit.ump`) imported `yaml`. Both `httpx` and `PyYAML` live in extras, not the core deps, so `import synapsekit` raised `ModuleNotFoundError`. These imports are now lazy (inside the methods that use them, matching the other optional-dep code paths), and a `tests/test_core_import_without_extras.py` guard imports the package in a fresh interpreter with every optional dependency blocked, so this bug class can't regress. Surfaced by the new core Docker image build.
- **`VoiceAgent`, `TextToSpeechTool`, and the Prolog symbolic backend no longer block the event loop** — each did synchronous filesystem writes (`mkdir`/`write_bytes`/`write_text`) directly inside an `async def`. The blocking IO is now offloaded via `asyncio.to_thread`. Also adds a CI gate (`scripts/check_async_blocking.py`) that fails the build on any blocking file/subprocess/sleep/network call sitting directly in a coroutine.
- **`pip install synapsekit[edge]` no longer pulls in `llama-cpp-python`** (#736) — the `edge` extra transitively reintroduced `llama-cpp-python`, whose `diskcache` dependency has no patched release for **CVE-2025-69872**, silently undoing the deliberately-emptied `llamacpp` extra. The `edge` extra now ships only `onnxruntime` + `sqlite-vec`; `LlamaCppLLM` remains available via an explicit `pip install llama-cpp-python` with an updated import-error message. Contributed by [@Abhay-Mmmm](https://github.com/Abhay-Mmmm)

- **`Neo4jWorldGraphBackend.query_subgraph` now filters nodes and documents by `min_confidence` / `as_of`, not just the returned edges** (#826) — a `neo4j` / `memgraph` graph backend previously returned entities (and their source documents) reachable only through low-confidence or time-expired edges, diverging from the in-memory backend on confidence-filtered and time-travel queries; `upsert_relation` now also records relation-provenance documents against both endpoints. Verified for parity against a live Neo4j via testcontainers.
- **`PGVectorStore` now works against real PostgreSQL** (#830) — previously broken end-to-end and only ever exercised with a `MagicMock`: it read a nonexistent `embeddings.dimension`, never called `register_vector`, never committed writes, used invalid index operator classes, and mishandled JSONB. Now derives the embedding dimension from the first vector (works with any embeddings backend), registers the pgvector adapters, uses autocommit and correct `vector_*_ops` opclasses (HNSW), and round-trips JSONB. Covered by real `pgvector/pgvector` testcontainers tests.
- **`RedisVectorStore` now works against real RediSearch** (#831) — previously raised `UnicodeDecodeError` on every search (it decoded the raw FLOAT32 embedding bytes returned by `FT.SEARCH`) and returned `[]` after a reconnect. Now scopes `FT.SEARCH` with a `RETURN` clause and no longer short-circuits on a missing cached dimension, so reconnected instances find persisted vectors. Covered by real `redis/redis-stack` testcontainers tests.
- **`ElasticsearchVectorStore` now works against a real Elasticsearch and modern clients** (#833) — it returned `[]` after a reconnect (short-circuited on a missing cached dim) and used the `body=` argument that is deprecated in elasticsearch-py 8.x and removed in 9.x. Now searches via the `knn=`/`mappings=` kwargs and finds persisted vectors on reconnect. Covered by real `elasticsearch:8.15` testcontainers tests.
- **`QdrantVectorStore` fixed against a real Qdrant** (#836) — it blocked the event loop (called the sync client directly inside `async def`), silently ignored `metadata_filter`, reused per-instance integer ids so a reconnect overwrote existing points, raised instead of returning `[]` before the first add, and used the deprecated `.search()` API. Now offloads via `asyncio.to_thread`, applies a native qdrant `Filter`, uses UUID ids (append-safe), returns `[]` for a missing collection, and uses `query_points`. Covered by real `qdrant/qdrant` testcontainers tests.
- **`WeaviateVectorStore` fixed against a real Weaviate (v4 client)** (#838) — it blocked the event loop, attached no vectors on insert (`insert_many` misuse), never created the collection (`collections.get` doesn't raise for missing), passed a list instead of `MetadataQuery`, scored every result 0.0, dropped metadata via `return_properties`, and omitted the required gRPC endpoint. Rewritten to offload via `asyncio.to_thread`, insert `DataObject(properties, vector)`, use `collections.exists`, derive score from cosine distance, apply a native `Filter`, and configure gRPC. Covered by real `weaviate` testcontainers tests.
- **`MilvusVectorStore` fixed against a real Milvus** (#840) — it blocked the event loop, built a schema with no primary key (rejected by Milvus), passed a plain dict where `create_index` needs an `IndexParams` object, and stored metadata as dynamic fields that never came back on search. Now offloads via `asyncio.to_thread`, declares an auto-id primary key, builds `IndexParams` via `prepare_index_params`, and stores/queries metadata as a JSON field. Covered by real Milvus Lite tests.
- **`CassandraVectorStore` fixed against real Cassandra 5.0** (#842) — the cassandra-driver path created the embedding column as `list<float>` with no SAI index (so `ORDER BY ... ANN OF` failed), hardcoded `score` to 1.0, and returned `[]` after a reconnect. Now creates a `vector<float, dim>` column plus a cosine `StorageAttachedIndex`, returns the real `similarity_cosine` score, and finds persisted rows on reconnect. Covered by real `cassandra:5` testcontainers tests.
- **`ClickHouseVectorStore` hardened against SQL injection and reconnect** (#844) — `top_k` was interpolated into the `LIMIT` clause unvalidated, and `search()` returned `[]` after a reconnect. Now `top_k` is cast to a positive int (a non-int payload raises `ValueError`) and searches check `EXISTS TABLE`, so a reconnected instance finds persisted rows. Covered by real `clickhouse-server` testcontainers tests.
- **`OpenSearchVectorStore` now finds persisted vectors on reconnect** (#846) — `search()` returned `[]` when it had no cached dimension, so a reconnected instance saw nothing. Now returns `[]` only on a genuine `NotFoundError`. Covered by real `opensearch` testcontainers tests.
- **`PostgresMemoryBackend` now reads records back from real Postgres** (#848) — it stored `embedding`/`metadata` as JSONB but never registered an asyncpg JSON codec, so every `fetch()` raised (`dict()`/`list()` on the raw JSON string). Now registers a JSONB codec and passes native list/dict, so records round-trip. Covered by real `postgres:16` testcontainers tests (and `RedisMemoryBackend` by real `redis:7` tests).

---

## [2.0.0] — 2026-07-15

Major release: the v2.0/v2.1 paradigm feature set plus a repo-wide production hardening pass. Includes breaking changes — see below.

### Breaking changes

- **`from synapsekit import AgentMemory` now returns the persistent memory class** (was the deprecated `AgentScratchpad` alias). The scratchpad is available as `AgentScratchpad`; `PersistentAgentMemory` remains a working alias for the persistent class.
- **Audit `verify()` without `trusted_keys` now returns `UNVERIFIABLE` instead of `MATCH`** — a self-signed bundle proves only internal consistency, not signer authenticity. Pass `trusted_keys={key_id: public_key}` for a real `MATCH`.
- **Audit bundle schema is now 1.2** (RFC 6962 Merkle construction and record-schema changes); bundles produced by earlier schema versions are not compatible with this verifier.
- **LLM `max_retries` now defaults to 2** (was 0) and retries are scoped to timeouts, connection errors, and 429/5xx.

### Added

- **`LivingMemory`** — file-routed agent memory that proposes diffable, signed patches instead of overwriting memory files directly; `MemoryFileRouter` categorizes content and resolves target paths; `MemoryPatch` / `FileDiffEngine` / `PatchStore` manage the propose, approve, apply, revert lifecycle; `MemoryPIIFilter` redacts sensitive content before it lands on disk; `OccurrenceTracker` for repeated-mention promotion; `synapsekit memory review / approve / reject / apply / revert / log` CLI; closes #741; contributed by [@DhruvGarg111](https://github.com/DhruvGarg111)
- **`VerifiableAgent`** — cryptographically verifiable audit and provenance system; hash-chained `AuditTracer` execution traces, RFC 6962 Merkle batch signing (Ed25519 default, pluggable BYOK/KMS), portable signed bundles via `export_audit_bundle`, a standalone `verify()` needing only stdlib + `cryptography` with a three-valued `MATCH` / `DRIFT` / `UNVERIFIABLE` verdict and `trusted_keys` pinning for real non-repudiation, replay, PII redaction before hashing, OTel/Prometheus wiring; `VerifiableAgent` proxy wrapper and `@audited` decorator; `synapsekit audit verify / replay` CLI; closes #738; contributed by [@Abhay-Mmmm](https://github.com/Abhay-Mmmm)
- **`GraphVectorStore`** — property graph RAG; graph-aware vector search fused with `PropertyGraphBackend` traversal (NetworkX and Neo4j backends), `KnowledgeGraphExtractor` for LLM or heuristic entity/relationship extraction, confidence/source/timestamp properties on extracted facts, `GraphMemoryBackend` / `GraphAgentMemory` wiring `AgentMemory(backend="graph", store=...)`; docs at `docs/rag/graph-rag.md`; closes #762; contributed by [@DhruvGarg111](https://github.com/DhruvGarg111)
- **`KnowledgeMesh`** — local-first personal knowledge mesh; SQLite-backed incremental indexing, offline hash embeddings, privacy filtering, cross-project entity resolution, duplicate detection; markdown/design-doc and local git repo loaders with heading hierarchy, file paths, line numbers, content hashes, commit subjects; optional Kuzu graph backend; `synapsekit mesh` CLI and `mesh_query` / `mesh_reindex` / `mesh_duplicates` / `mesh_status` MCP tools; closes #740; contributed by [@DhruvGarg111](https://github.com/DhruvGarg111)
- **`SelfImprovingAgent`** — eval-gated agent config evolution; observes `FeedbackCollector` traces, proposes signed `AgentConfigPatch` diffs, validates prompt candidates with `EvalSuite` / `PromptOptimizer`, and canaries accepted changes through `AutoRolloutManager`; patches are eval-blocked by default and reversible via `agent.rollback(patch_id)`; audit trail via `agent.evolution_history()` or `synapsekit agent inspect-evolution <agent-id>`; `AgentConfigPatch`, `AgentConfigSnapshot`, `AgentEvolutionAuditLog`, `MetaAnalyzer`, `EvalSuite` exported at top level; closes #732; contributed by [@DhruvGarg111](https://github.com/DhruvGarg111)
- **`AgentSwarm`** — market-based agent routing on top of `AgentFederation`; agents bid with estimated cost, quality, and confidence; `MarketPolicy` supports sealed-bid, Vickrey, English, multi-winner, and coalition auctions; `Reputation` tracks per-agent per-task-category outcomes with UCB and Thompson-sampling exploration; deterministic runs via `seed=42`; `RoutingStrategy.MARKET` integrates with existing `AgentFederation.run()`; `pip install synapsekit[redis]` for Redis-backed reputation; closes #734; contributed by [@DhruvGarg111](https://github.com/DhruvGarg111)
- **`ComputerUseAgent`** — provider-agnostic screen observation and action execution; normalises Anthropic, OpenAI, and open-source computer-use providers to a unified `ComputerAction` schema; `SafetyPolicy` enforces forbidden apps, domain allow/block lists, keyword confirmation triggers, and PII-in-text detection; `@requires_human_confirmation` decorator; JSONL session recording with `RecordedSession` replay; `BrowserScreenProvider` (Playwright) and `PyAutoGUIScreenProvider` adapters; `pip install synapsekit[computer-use]`; closes #737; contributed by [@DhruvGarg111](https://github.com/DhruvGarg111)
- **`NeuroSymbolicAgent`** — neuro-symbolic reasoning layer; LLM extracts formal constraints, symbolic backends solve/verify them, callers receive structured `ProofTrace` metadata; optional `Z3Backend`, `SympyBackend`, `MiniZincBackend`, `PrologBackend`; `@verified_tool` decorator gates tool execution on proof success; `on_unverified` policy (`"retry"`, `"reject"`, `"flag"`); `pip install synapsekit[symbolic]`; closes #733; contributed by [@DhruvGarg111](https://github.com/DhruvGarg111)
- **`WorldModelRAG`** — temporal knowledge-graph RAG; LLM or heuristic extraction builds a world model of entities, relations, events, and causal links; `InMemoryWorldGraphBackend` stores nodes/edges with temporal validity windows; `HybridWorldModelRetriever` fuses graph and vector results with reciprocal rank fusion; `CausalLinker` scores candidate edges; Mermaid subgraph export; closes #735; contributed by [@DhruvGarg111](https://github.com/DhruvGarg111)
- **`EdgeRuntime`** — local-first inference with explicit policy-gated cloud fallback; routes locally by default; `FallbackPolicy` configures cloud escalation on context overflow, unsupported tools, user opt-in, or local errors; PII redaction before cloud fallback via existing `PIIRedactor` path; `ONNXEmbeddings` and `MLXLLM` providers behind `synapsekit[onnx]` / `synapsekit[mlx]` extras; `synapsekit edge list / pull / quantize` CLI commands; closes #736; contributed by [@DhruvGarg111](https://github.com/DhruvGarg111)
- **Comparison matrix** — `docs/comparison-matrix.md` with sourced feature comparison against LangChain, LlamaIndex, Haystack, and DSPy; quarterly update workflow in `.github/workflows/comparison-matrix-update.yml`; closes #670; contributed by [@DhruvGarg111](https://github.com/DhruvGarg111)

### Fixed

#### Production hardening audit

A six-dimension audit (architecture, performance, security, reliability, packaging, correctness) found and fixed 42 issues (#773–#814), each shipped with a regression test.

**Behavior changes (note before upgrading):**

- **`from synapsekit import AgentMemory` now returns the persistent memory class** (#790) — previously it returned the deprecated `AgentScratchpad` alias and warned on every use. The scratchpad is still available as `AgentScratchpad`, and `PersistentAgentMemory` remains a working alias for the persistent class.
- **Audit `verify()` without `trusted_keys` now returns `UNVERIFIABLE`, not `MATCH`** (#811) — a self-signed bundle proves only internal consistency, not signer authenticity, so an unpinned verify no longer reports `MATCH`. Pass `trusted_keys={key_id: public_key}` for a real `MATCH` verdict.

**Security:**

- **CalculatorTool `eval()` sandbox escape / RCE** (#801) — a default agent tool ran `eval()` with no AST validation, allowing `().__class__.__bases__[0].__subclasses__()` escapes to `os.system` and a `9**9**9` DoS; replaced with an AST allowlist and an exponent bound
- **ShellTool allowlist bypass on Windows** (#802) — the allowlist checked `argv[0]` but ran the raw string via `shell=True`, so `echo hi & curl evil` bypassed it; now always runs parsed argv without a shell and rejects metacharacters when an allowlist is set
- **SitemapLoader and WebLoader SSRF** (#806, #807) — added a shared fail-closed URL guard that blocks private/loopback/link-local/reserved and cloud-metadata (169.254.169.254) addresses, re-validates on redirects, and rejects on DNS-resolution failure (WebLoader previously failed open); SitemapLoader now validates every discovered and nested-index URL
- **BrowserTool SSRF** (#803) — `allowed_domains=None` allowed all hosts with no private-IP guard; added an `ipaddress`-based private/loopback/link-local guard applied regardless of the allowlist and re-checked on redirects, with an `allow_private_ips` opt-out
- **ConfigLoader leaked embedded credentials** (#808) — redaction keyed only on config key names, emitting `DATABASE_URL=postgres://user:pass@host` verbatim; now also redacts credentials embedded in values (URL userinfo, known key prefixes, high-entropy tokens)
- **Cypher injection via `max_hops` in the Neo4j KG backend** (#783) — an unchecked value was interpolated into a variable-length relationship pattern; now cast to a bounded int
- **Cassandra CQL built by f-string** (#784) — `top_k` is now validated as a positive int and bound as a parameter
- **Audit verifier trust anchor** (#811, see behavior changes above)

**Correctness:**

- **`BaseTool.parameters` was a `dataclasses.Field` object, not a dict** (#799) — `field(default_factory=dict)` used outside a dataclass made `tool.schema()` unserializable and crashed `FunctionCallingAgent` on the first API call for any tool without explicit parameters; replaced with a property returning a fresh dict
- **`@audited` turned task cancellation into `UnboundLocalError`** (#809) — `emitted_kind` was unset on the `CancelledError` path; now initialized before the try and a record is written on cancellation
- **`@audited` mishandled async-generator (streaming) methods** (#810) — streaming methods fell into the sync wrapper and recorded a non-deterministic `<async_generator at 0x…>` repr before any token; added an async-gen branch that records a deterministic aggregate and mid-stream errors
- **Token totals recorded as cumulative values** (#791) — `RAGPipeline.stream` and `SelfHealingRAG` passed the running `tokens_used` total to the tracer, inflating cost reports quadratically; now record the per-call delta
- **LivingMemory auto-apply lost earlier patches** (#792) — proposals computed diffs from a stale one-time snapshot, so a second patch to a file overwrote the first; the snapshot is now updated after each applied patch
- **ReAct `Action Input` regex captured the rest of the completion** (#800) — `re.DOTALL` swallowed subsequent blocks into the tool input; now stops at the next section marker
- **`VectorStore.search(top_k=0)` returned all documents** (#780) — `np.argpartition(scores, -0)[-0:]` sliced the whole array; now early-returns `[]`
- **Text splitters silently dropped data on bad overlap** (#781) — `chunk_overlap >= chunk_size` raised or returned `[]` (silent no-op ingest); now validated in `__init__`
- **`top_k=0` conflated with unset** across RAG, retrieval, and mesh entry points (#782, #793, #814) — `top_k or default` replaced with `is not None` checks
- **HuggingFaceLLM overrode `temperature=0.0` / `max_tokens=0`** (#773) — falsy-zero defaults replaced with `is not None`
- **LLM response cache key omitted `system_prompt` and provider** (#774) — instances sharing a cache path but different system prompts returned each other's answers; both are now folded into the key
- **AgentMemory overflow consolidation degenerated to one-episode summaries** (#794) — now consolidates `max(overflow, consolidation_window)` episodes
- **RAG pipeline shared one metadata dict across all chunks** (#795) — mutating one result's metadata mutated every chunk; each chunk now gets an independent dict
- **HybridSearch dropped prior docs on incremental add** (#786) — `add_documents` replaced the index instead of extending it
- **AgentMemory consolidation failures were swallowed silently** (#798) — now logged at warning level

**Reliability:**

- **LLM retries were off by default with no timeout knob** (#775) — `max_retries` now defaults to 2 and `LLMConfig` gained a `timeout` field
- **Retry classification used substring matching** (#776) — now classifies on exception type / HTTP status (retries only timeouts, connection errors, 429/5xx), adds jitter, and honors `Retry-After`
- **Sync Redis and SQLite cache calls blocked the event loop** (#777, #778) — async paths offload to a thread; SQLite cache uses `check_same_thread=False` + WAL + busy timeout + a lock; Redis gets socket timeouts
- **`LLMConfig` was unvalidated** (#779) — added `__post_init__` range validation for temperature, top_p, max_tokens, and timeout
- **AgentFederation had no failover** (#804) — `run()` now wraps calls in `asyncio.wait_for` and fails over to the next healthy candidate
- **HybridSearch failed entirely when the vector store was down** (#787) — now falls back to BM25-only results

**Performance:**

- **`WorldModelRAG.ingest` embedded one document at a time** (#788) — now batches all documents into a single embedding call
- **`AgentMemory.recall` scored cosine in pure Python over every row** (#796) — vectorized with a single numpy matrix product
- **AgentMemory SQLite backend opened a connection per operation** (#797) — now uses one persistent WAL connection and batches per-hit `touch` updates
- **KnowledgeMesh blocked the event loop and full-scanned SQLite** (#812, #813, #814) — reindex/query I/O moved to threads, added an index on `mesh_chunks(path, active)`, and replaced whole-set materialization with a candidate-scoped query
- **RAG-fusion ran query variants sequentially** (#785) — now fans out with `asyncio.gather`
- **HybridSearch full-sorted the corpus per query** (#786) — now uses `heapq.nlargest`
- **Blocking file/SQLite I/O in agent tools** (#805) — `sql_query`, `file_read`, and `file_write` moved to threads; file tools gained a max-bytes cap

**Packaging:**

- **`synapsekit[cassandra]` did not install `astrapy`** (#789) — the extra pointed users at an install that still failed; `astrapy` added to the extra and the error message corrected

- **`PatchStore.update()` clobbered signatures** — every state transition re-signed the patch with an empty secret right after the caller had already signed it correctly, breaking `verify()` for applied/reverted patches; removed the redundant sign call
- **`MemoryPIIFilter` missed API keys** — the clean-content check only looked at `PIIDetector`'s small pattern set, so content containing only an API key or secret slipped through unredacted; the check now also covers the filter's own redaction patterns
- **`MemoryPIIFilter` violation type parsing** — `redact=False` mode mis-parsed violation strings like `"PII detected (email): 2 instance(s)"` into `"email)"` instead of `"email"`; replaced with a proper regex parse
- **`CreditCardDetector` over-redaction** — matched any 13-16 digit run with no structural check, redacting ordinary numeric IDs and not actually verifying card numbers; added a Luhn checksum gate
- **`AuditRecord.from_dict` skipped freezing** — records reconstructed from a loaded bundle were mutable while records built via `AuditTracer.record` were frozen; both paths now freeze consistently
- **`verify_chain` had no run boundary** — an unfiltered multi-run record list could validate a forged cross-run chain link; now raises if records span more than one `run_id`
- **Canonical JSON hashing ignored Unicode normalization** — combining and precomposed forms of the same string hashed differently; added NFC normalization before hashing
- **Merkle leaves lacked domain separation** — only internal nodes carried the RFC 6962 domain prefix; added the leaf prefix so both match the spec
- **Neo4j `traverse` used a query parameter for the hop bound** — Cypher requires variable-length relationship bounds as integer literals, so any non-default `max_hops` threw `CypherSyntaxError`; now interpolates a validated int
- **`GraphMemoryBackend.delete` leaked graph state** — removed the record but left its node and edges behind in the graph store; now removes both
- **`GraphVectorStore` cross-tenant metadata leak** — graph traversal results were fused into search metadata before the tenant filter was applied, letting a filtered search surface other tenants' entities and relations; filtering now happens before fusion
- **`_seed_ids` linear scan** — scanned every graph node per vector hit instead of using an index; added an inverted document-to-node index for O(1) lookup
- **Empty `active_chunk_ids` treated as no filter** — an emptied or pre-first-reindex mesh could still surface stale or deleted content; empty sets now correctly yield zero hits
- **Mesh reindex leaked stale vectors** — changed files got new vectors but the old ones were never removed from the vector store, only flagged inactive in SQLite; reindex now deletes the stale vectors too
- **`mesh stop` did not stop the real daemon** — it only wrote a status row and never signaled the actually running watch process; now sends a real `SIGTERM` to the daemon's PID
- **Mesh default include pattern missed top-level docs** — `docs/**/*.md` never matched files directly under `docs/`, only masked by a co-listed `*.md` fallback; fixed the glob matcher's globstar handling
- **Vickrey auction settlement** — `AgentSwarm` was computing settlement price from the second-highest-*score* agent's cost instead of the second-highest *bid price*; corrected to sort by `estimated_cost` descending before picking the second entry
- **Swarm winner tie-breaking** — `_select_winners` sort was non-deterministic when scores and costs were equal; `agent_id` added as final tiebreaker
- **Coalition underflow** — `CoalitionFormer.form()` silently returned a 1-element list when fewer agents than `max_size` were available; now raises `LookupError`
- **Zero-cost agent reputation** — `_synthetic_bid` used `avg_cost or cost_multiplier` (truthiness check); `0.0` cost was falsy and always fell back to the multiplier; `ReputationSnapshot.avg_cost` changed to `float | None`
- **`@requires_human_confirmation` broke async functions** — decorator always returned a sync wrapper, making decorated `async def` functions lose their coroutine identity; now dispatches an `async def` wrapper for coroutine functions
- **`SafetyPolicy.confirm_before=[]`** — empty list was falsy, causing it to fall through to the default `_DANGEROUS_WORDS` keyword list even when the caller explicitly passed an empty list to disable keyword confirmation; fixed with `is not None` guard
- **`EdgeRuntime.stream` error recovery** — on local error the fallback text was yielded character-by-character (`for token in str`); fixed to `yield fallback_str` as a single chunk
- **`SympyBackend` timeout non-functional** — inner `async def _solve()` called blocking `eval()` without yielding; `asyncio.wait_for` cannot cancel a running C-extension; moved `eval()` to `asyncio.to_thread()` matching the `Z3Backend` pattern
- **`_parse_json_objects` backtick stripping** — `cleaned.strip("\`")` stripped all leading/trailing backtick characters rather than just the triple-backtick fence; replaced with `[3:]` / `[:-3]` slicing
- **`AgentConfigPatch.to_dict()` side effect** — was calling `self.sign()` as part of serialisation, mutating the patch on every `to_dict()` call; removed auto-sign
- **`_run_agent_async` missing `iscoroutinefunction` guard** — the `arun` branch assumed any `arun` attribute was a coroutine; added `inspect.iscoroutinefunction(agent.arun)` guard
- **`ConstraintExtractor` bare `JSONDecodeError`** — fallback regex `json.loads` call was not wrapped; now raises `VerificationFailure` with a clear message

### Docs

- **Architecture deep dive** — long-form internals doc covering async runtime model, RAG flow, graph engine, agent loop, plugin system, and extension points; aimed at contributors and power users; contributed by [@DhruvGarg111](https://github.com/DhruvGarg111); closes #673

### CI

- **Discord label routing** — new `.github/workflows/discord-labels.yml` posts to a per-label Discord channel whenever an issue or PR is opened, labeled, or reopened; 14 topic and release-milestone channels; missing secrets are skipped gracefully; closes #753

---

## [1.9.1] — 2026-05-27

### Fixed

- **`__version__` mismatch** — `synapsekit.__version__` was left at `1.7.0` after the v1.8.0 and v1.9.0 releases, causing `test_version_matches_pyproject` to fail on a fresh clone; bumped to `1.9.0` to match `pyproject.toml`
- **`uv.lock` out of sync** — `prometheus-client` was added to the `observe` extra in v1.9.0 but the regenerated lock file was never committed; committed now so `uv sync` no longer produces a dirty diff
- **Voice lazy imports** — `from .voice import …` in `__init__.py` was unconditional, meaning importing `synapsekit` always pulled in the full voice module tree; any future top-level dep in voice code would silently break `import synapsekit` for users without voice extras; all 17 voice exports (`VoicePipeline`, STT/TTS/VAD providers, types) moved to `_LAZY_IMPORTS` so they only load on first access; closes #720
- **Ollama async stream** — `AsyncClient.chat()` returns an `AsyncIterator` but was not being awaited before iteration, causing `TypeError: 'async for' requires an object with __aiter__ method, got coroutine`; fixed by awaiting the call in `stream_with_messages`; closes #721

---

## [1.9.0] — 2026-05-20

### Added

- **`SmartContextManager`** — hierarchical context window management with Anthropic prompt caching; static system prompt and running summary both tagged with `cache_control: ephemeral` for up to 80% cost reduction on repeated calls; sliding window pruning summarises older messages via a configurable cheap LLM; search results slot bounded by token budget; `cache_control` keys stripped automatically for non-Anthropic providers; `synapsekit.memory.SmartContextManager`; closes #697
- **`PrometheusMetrics`** — Prometheus metrics exporter for LLM observability; records `synapsekit_cost_usd_total`, `synapsekit_tokens_total`, and `synapsekit_latency_seconds` (histogram) per model/provider; integrates with the existing `observe` span pipeline via `record_span()`; degrades silently when `prometheus-client` is not installed; `pip install synapsekit[observe]` now installs `prometheus-client>=0.20`; includes full Helm chart for Prometheus + Grafana deployment in `assets/helm/synapsekit-observability/`; closes #695
- **`StructuredOutput`** — provider-agnostic structured output helper with Pydantic v2 validation, configurable retry strategy, exponential backoff, fallback provider/model support, and full streaming via `IncrementalJSONBuffer` (detects JSON completion during stream); `StructuredOutputRetryStrategy` controls max attempts, backoff, and fallback activation; all attempts recorded in `StructuredOutputResult.attempts` metadata; streaming yields `chunk`, `retry`, and `result` events; `StructuredOutput`, `StructuredOutputRetryStrategy`, `IncrementalJSONBuffer`, and friends exported at top level; closes #696
- **`AgentFederation` + `AgentRegistry`** — distributed agent registry and federation layer; `InMemoryAgentRegistry` and `RedisAgentRegistry` store `AgentMetadata` with heartbeat-based health checks and stale-agent pruning; `AgentFederation` routes prompts across registered agents using round-robin, capacity-aware, or cost-aware strategies (`RoutingStrategy`); tag and tool-based discovery filters; `LocalAgentClient` for in-process agents; `RedisAgentRegistry` requires `synapsekit[redis]`; closes #701
- **`ContinuousTrainer` fine-tuning pipeline** — closed-loop continuous fine-tuning with production feedback; `FeedbackCollector` batches samples via async queue with pluggable backends (in-memory default); `TrainingDataGenerator` exports JSONL with preference pairs; `OpenAIFineTuneProvider` and `AnthropicFineTuneProvider` submit and poll jobs; `ABTestRouter` sticky-routes traffic by SHA-256 user bucket; `AutoRolloutManager` stages rollout (5 → 25 → 50 → 100%) with latency/cost/quality regression rollback; `CostBenefitAnalyzer` projects ROI and payback days; `ContinuousTrainer` orchestrator wires all components; `pip install synapsekit[training]` for OpenAI fine-tune SDK; closes #702
- **Benchmark harness** — `benchmarks/` directory with `pytest-benchmark` config, percentile report script, and ASV config for regression tracking; `make bench` and `make bench-compare` targets; CI workflow in `.github/workflows/benchmarks.yml` runs on version tags and nightly; `pip install synapsekit[bench]` for `pytest-benchmark`, `asv`, and `psutil`; closes #626

---

## [1.8.0] — 2026-05-17

### Added

- **`KnowledgeGraphBuilder` + `KGRetriever` + `HybridKGRetriever`** — multi-hop knowledge graph retrieval for entity-relationship queries across documents; NetworkX (in-memory) and Neo4j backends; `KnowledgeGraphBuilder` extracts entities and triples with LLM; `KGRetriever` traverses graph with depth-first search; `HybridKGRetriever` merges vector + graph results; wired into `RAG` facade via optional `graph_store` parameter; `pip install synapsekit[graph]` for NetworkX/Neo4j; closes #699
- **`RAGEvaluator`** — production sampled RAG quality judge with LLM-based scoring, alert sinks (Slack, PagerDuty, email), ROI metrics, and remediation suggestions; non-blocking evaluation doesn't interrupt main RAG path; deterministic sampling via hash-based stratification; cost tracking and dashboard integration via `TokenTracer`; exposes per-metric thresholds, average scores, and alert history; closes #698
- **`ReasoningAgent`** — intelligent routing for reasoning models with complexity classifier (LLM + heuristic paths), thinking token budget enforcement, timeout fallback to fast LLM, and cost tracking; supports o1/o3, Claude thinking, Gemini thinking, DeepSeek R1, Qwen QwQ; extends `ReasoningLLM` with native tool-calling for OpenAI/Anthropic; `ReasoningAgentConfig` with configurable complexity thresholds and budgets; closes #700
- **VoicePipeline performance enhancements** — PiperTTS instance caching eliminates per-sentence model reload from disk (critical fix); DeepgramSTT client caching prevents TLS handshake per utterance; exported BaseVAD, BaseSTT, BaseTTS, AudioFrame, TranscriptChunk for custom voice provider authors; comprehensive `voice_assistant.py` example with persistent memory across sessions; `pip install synapsekit[voice-piper]` new extra; closes #512
- **`FallbackChain` empty response handling** — explicit `fallback_on_empty` configuration for `FallbackChainConfig` allows chains to accept empty first responses when intended instead of always falling back; default `True` preserves backward compatibility; closes #703

### Changed

- **`FederatedRetriever` input coercion** — `_coerce_result` / `_coerce_results` helpers replace scattered inline list comprehensions in `_fetch_local` and `_fetch_remote`; now handles `str`, `dict` (with optional nested `document` object), any object with a `.text` attribute, and `(text, score)` tuples uniformly; fixes a `dict | None = {}` type annotation inconsistency and eliminates a double `item.get("metadata")` lookup

### Fixed

- **`tzdata` on Windows** — added `tzdata>=2024.1` to the `cron` optional extra and dev dependency group so `croniter` resolves timezone data correctly on Windows (where the OS tzdb is absent)

---

## [1.7.0] — 2026-05-06

### Added

- **`ReasoningLLM`** — unified adapter for reasoning/chain-of-thought models; auto-detects provider from model name (`o1`/`o3` → OpenAI, `claude` → Anthropic, `gemini` → Google, `deepseek` → DeepSeek, `qwq` → Qwen); `agenerate()` returns a `ReasoningResponse` dataclass with `answer`, `thinking`, `thinking_tokens`, `answer_tokens`, `total_tokens`, `model`, and `provider`; `astream()` yields `ReasoningStreamChunk` objects with `text` and `is_thinking` and enforces reasoning-before-answer ordering; each backend in `llm/providers/` handles provider-specific API params (reasoning_effort, thinking blocks, thinkingConfig, reasoning_content)
- **`CostQualityRouter`** — learning-based LLM router with explore/exploit strategy; round-robins across candidates for `explore_n` calls then routes to the cheapest model meeting `quality_threshold`; respects `budget_per_call_usd` when set; `stats()` returns per-model averages and the Pareto frontier of cost vs quality; optional `eval_suite` parameter for quality scoring via an existing `@eval_case` suite
- **`PromptOptimizer`** — scores prompt variants against an `@eval_case` suite and returns the best `PromptCandidate`; variants can be provided directly or generated by an LLM; stops early when `budget_usd` is exceeded; all evaluated candidates stored in `.candidates` sorted by score descending; `PromptVariantRunner` handles per-variant execution with async/sync case support and optional prompt injection

### Performance

- **`orjson` fast JSON serialization** — `_json.py` wrapper tries `orjson`, falls back to stdlib `json`; wired into all hot paths: LLM cache key generation, filesystem cache read/write, vector store save/load, all graph checkpointers (SQLite, JSON file, Redis, Postgres), and JSON output parser
- **`uvloop` fast event loop** — `_loop.py` installs `uvloop` if available; called at module load (`_compat.py`) and CLI entry (`cli/main.py`)
- **`xxhash` fast cache key hashing** — cache key generation uses Rust BLAKE3 → xxhash → sha256 fallback chain (5-10x faster than sha256-only); `dumps_bytes()` avoids redundant encode step
- **Pre-allocated vector buffer** — `InMemoryVectorStore` uses a doubling buffer strategy instead of `np.vstack` on every consolidation, eliminating O(n) array copies per search
- **Vectorised MMR** — `search_mmr()` greedy selection loop fully vectorised with numpy masked arrays; replaces Python-level inner `for` loop with single `np.argmax` + `np.maximum` per selection round
- **`__slots__`** on hot classes — `AsyncLRUCache`, `FilesystemLLMCache`, `RecursiveCharacterTextSplitter`, `CharacterTextSplitter`, `JSONParser` (~30% less per-instance memory)
- **Lazy imports in `_compat.py`** — `asyncio` and `collections.abc` imported inside `run_sync()` instead of at module level for faster import time
- **`performance` optional extra** — `pip install synapsekit[performance]` installs `orjson`, `uvloop` (non-Windows), and `xxhash`
- **Rust extension module** (`_rust/`) — optional PyO3 crate with text chunking (`recursive_split`, `character_split` with recursive sub-splitting and UTF-8 char boundary safety), cache key hashing (`fast_cache_key` using canonical JSON via `_json.dumps_bytes` + xxh3_128 — keys match Python xxhash path exactly), and batch metadata serialization (`serialize_metadata_list`/`deserialize_metadata_list` via native serde_json ↔ Python object conversion); pure-Python fallback for all functions; 15 Rust unit tests + 11 Python parity tests; build with `maturin develop`

### Added

- **`FederatedRetriever`** — fan-out retrieval across multiple local and remote sources in parallel; merges results using RRF (default), normalised score fusion, or round-robin interleave; per-source timeout with graceful partial results on failure; near-duplicate deduplication via `SequenceMatcher`; optional bearer-token auth for remote HTTP endpoints; closes #595
- **Discord `#stats` workflow** — GitHub Actions workflow that patches a pinned message in the Discord `#stats` channel every 6 hours with live GitHub stars, forks, open issues, 30-day PyPI downloads, and latest version; requires `DISCORD_STATS_WEBHOOK` and `DISCORD_STATS_MESSAGE_ID` secrets; zero external deps (stdlib `urllib`)

- **Fine-tune data flywheel** (closes #515) — end-to-end workflow from eval results to fine-tuned model; `@eval_case(capture_io=True)` captures `input`/`output`/`ideal` in snapshots; `EvalDataset` loads, filters, and exports to OpenAI, Anthropic, Together AI, and DPO formats; `FineTuner` submits jobs to OpenAI and Together AI and polls via `wait()`; CLI commands `synapsekit eval report/export/compare` and `synapsekit finetune submit/status/wait`; `examples/fine_tune_flywheel.py` end-to-end example

### Fixed

- **README Integrations icons** — replaced all `cdn.simpleicons.org` URLs with Google Favicons (`google.com/s2/favicons`) after GitHub's image proxy (camo) was found to block the SimpleIcons CDN; all 60+ integration icons now load reliably across light and dark themes
- **v1.7.0 production test suite** — fixed 24 API-mismatch failures in `tests/test_v170_production.py`: corrected `AsyncLRUCache.make_key()` usage (was importing non-existent `make_cache_key`), `put()`/`get()` are synchronous (removed erroneous `await`), splitter method is `.split()` not `.split_text()`, `__slots__` check uses `type(s).__dict__`, `InMemoryVectorStore` uses async `add()`/`search()`/`search_mmr()` with `SynapsekitEmbeddings` backend (not LangChain-style API), reasoning provider public attrs (`model`/`thinking`/`provider`) and method names (`generate()`/`stream()` not `agenerate()`/`astream()`), `qwq` maps to `"qwen"` provider, `PromptOptimizer.run()` takes `instructions` (variants go to constructor), `FederatedRetriever` exceptions swallowed by `asyncio.gather(return_exceptions=True)` return empty lists, and dedup threshold semantics corrected (higher threshold = more permissive)

- **Type annotation suppressions for Rust/optional-import fallback paths** — added `# type: ignore` to `_json.py`, `_cache.py`, `text_splitters/character.py`, and `text_splitters/recursive.py` where mypy cannot narrow module-level callables set to `None` in `except ImportError` blocks; no behaviour change

- **`SelfHealingRAG`** — retry-on-low-faithfulness RAG wrapper; cycles through a list of `RetrievalStrategy` implementations until the `FaithfulnessMetric` score meets `quality_threshold`; exposes `last_report` (`SelfHealingReport`) with attempt count, retry count, per-attempt scores, and winning strategy name; `ask_sync()` for synchronous callers; `max_retries` bounds total attempts; gracefully falls back to best answer when all strategies are exhausted
- **`ContextPacker`** — token-budget-aware chunk packing for long-context models; three ranking strategies (`relevance`, `recency`, `diversity`); near-duplicate deduplication via `SequenceMatcher` with configurable `dedup_threshold`; `lost-in-middle` and `as-is` ordering; returns structured dicts with `text`, `score`, `metadata`, and `token_count`; accepts raw strings, `Document` objects, and scored dicts from any retriever
- **`FullContextRetriever`** — retriever wrapper that auto-switches between full-document ingestion and chunked ingestion based on a per-document token count; exposes `add_document()` so `RAGPipeline` can route large vs. small documents without caller changes; delegates `retrieve()` and `retrieve_with_scores()` to the wrapped retriever
- **`TokenCounter`** — model-aware token counting with `tiktoken`, `transformers`, or custom callable backends; `auto` mode tries tiktoken first; 8 192-entry LRU cache via `count_cached()`; `pip install synapsekit[tiktoken]` or `pip install synapsekit[transformers]`
- **EvalHub community registry + `synapsekit bench` CLI** — `synapsekit bench --list` shows available community suites; `synapsekit bench --suite community/rag-general --model gpt-4o-mini` runs evaluation and prints table or JSON results vs. community baseline; `synapsekit bench --publish my_evals/ --name myorg/rag-finance` packages and submits a PR to add a new suite; 5 bundled suites: `community/customer-support` (20 cases), `community/code-generation` (30 cases), `community/rag-general` (25 cases), `community/summarization` (15 cases), `community/qa-hotpotqa` (50 cases)
- **Source-aware RAG context** — `RAGPipeline.stream()` injects `[SOURCE]` blocks with `source_type`, `source`, `chunk_type`, `page`, `timestamp`, `locator`, and `score` metadata above each `<document>` block when the retriever returns scored results; enables accurate in-answer citations
- **Unified media locator metadata** — `AudioLoader`, `VideoLoader`, `ImageLoader`, and `PDFLoader` now emit consistent `locator`, `chunk_type`, `media_type`, and `timestamp` fields in document metadata; `PDFLoader` adds `aload()` async wrapper; `RAG.add()` facade routes PDF files via MIME type in addition to the existing audio/video/image routing
- **`loaders/_media_utils.py`** — shared `format_locator()` and `format_seconds()` helpers extracted from `AudioLoader` and `VideoLoader` to eliminate duplication

### Fixed

- **Security dep bumps** — pinned `pillow>=12.2.0`, `pyasn1>=0.6.3` (high-severity Dependabot advisories); bumped `lxml>=6.1.0` (html/dev extras) and `gitpython>=3.1.47` (git extra) to pick up upstream CVE fixes

---

## [1.6.0] — 2026-04-26

### Added

- **`CronTrigger`** — schedule-based agent execution via cron expressions (`"0 9 * * *"`) or interval shorthand (`"30m"`, `"1h"`, `"2d"`); missed-run policies (`skip` / `catch_up` with configurable `max_catch_up_runs`); injectable `clock` and `sleep_func` for deterministic testing; `AuditLog` integration; `TriggerResult` dataclass with timing and error metadata; optional `result_sink` callback (sync or async); `pip install synapsekit[cron]` for cron expression support; closes #583
- **`SimpleAgent` + `agent()` factory** — 10-line happy-path agent facade wrapping `AgentExecutor`; optional conversation memory; auto-detects LLM provider from model name prefix; `agent(model="gpt-4o-mini", api_key="...")` returns a `SimpleAgent` ready to call `.run()` or `.arun()`; `api_key` defaults to `""` so local models (Ollama, LMStudio) require no key; closes #599
- **`make_llm()` shared factory** (`llm/_factory.py`) — extracted provider-detection and `LLMConfig` construction into a reusable module; resolves provider from model name prefix (claude→anthropic, gemini→google, llama→groq, `/`→openrouter, etc.); used by `SimpleAgent` and available as a public helper
- **Auto-eval metrics on traced RAG calls** — `RAGPipeline` with `auto_eval=True` and an active `TokenTracer` fires a background `asyncio.Task` after every streamed answer that evaluates faithfulness and relevancy via `EvaluationPipeline`; scores are recorded in `TokenTracer` via `record_quality()`; `tracer.summary()` now includes `avg_faithfulness`, `avg_relevancy` (both `None` when no data), and `quality_trend` (`improving`/`degrading`/`stable`); `RAGPipeline.wait_for_auto_eval()` flushes pending tasks; closes #591
- **`PubMedLoader`** — fetch medical literature from NCBI PubMed via E-utilities API (`esearch` + `efetch`); supports search query string, `max_results`, `api_key` for higher rate limits, and `email` for NCBI courtesy identification; parses XML response; returns one `Document` per article with title, abstract, authors, and PMID in metadata; async `aload()` via executor; no extra deps; closes #75
- **`SnowflakeLoader`** — load documents from a Snowflake query; configurable `account`, `user`, `password`, `database`, `schema`, `warehouse`, `role`; `text_fields` selects which columns form the document body; `limit` pushed into SQL via `LIMIT N` (not Python slice) so Snowflake enforces it server-side; async `aload()` via executor; `pip install synapsekit[snowflake]`; closes #90
- **`ImageGenerationTool`** — generate images from text prompts via OpenAI DALL-E 3; subclasses `BaseTool`; lazy `AsyncOpenAI` client initialisation; returns `ToolResult` with image URL; validates `size` (1024×1024, 1792×1024, 1024×1792) and `quality` (standard, hd) before calling API; exported from `synapsekit.agents.tools` and top-level `synapsekit`; `pip install openai`; closes #215
- **`AgentMemory`** — persistent episodic + semantic memory for agents; SQLite, Redis, Postgres, and in-memory backends; semantic similarity search via cosine over optional embeddings or built-in bloom hash; episodic consolidation window; integrates into `ReactAgent`, `FunctionCallingAgent`, and graph `llm_node`; closes #506
- **`BrowserTool`** — Playwright-based browser automation tool; selector-based interaction (navigate, click, fill, get_text, screenshot, etc.); domain allow/block lists; optional screenshot-on-action for multimodal models; `pip install synapsekit[browser]`; closes #555
- **`MongoDBAtlasVectorStore`** — MongoDB Atlas Vector Search backend; MQL metadata filter passthrough; configurable vector/text/metadata field names; `pip install synapsekit[mongodb-vector]`; closes #121
- **Multimodal loaders enhanced** — `AudioLoader` gains Whisper transcription (API + local); `VideoLoader` gains frame extraction + multi-track audio; `ImageLoader` enhanced with vision-model captioning; `RAG.add()` facade auto-routes by MIME type; closes #510
- **`YouTubeLoader`** — load video transcripts via `youtube-transcript-api`; accepts full URLs or bare video IDs; language override; `pip install synapsekit[youtube_transcript]`; closes #560
- **`ObsidianLoader`** — load Obsidian vault Markdown notes; extracts YAML frontmatter, wikilinks, and tags into metadata; recursive vault traversal; no extra deps; closes #562
- **`AirtableLoader`** — load Airtable records via `pyairtable`; configurable `text_fields` and `metadata_fields`; `pip install synapsekit[airtable]`; closes #561
- **`SitemapLoader`** — recursive sitemap XML parsing with HTTP page fetch; BS4 text extraction; configurable max depth and concurrency; `pip install synapsekit[sitemap]` (uses beautifulsoup4); closes #557
- **`HubSpotLoader`** — load HubSpot CRM contacts, deals, and tickets via `hubspot-api-client`; configurable `text_fields` and `metadata_fields`; `pip install synapsekit[hubspot]`; closes #88
- **`SalesforceLoader`** — load Salesforce records via SOQL query using `simple_salesforce`; configurable field mappings; `pip install synapsekit[salesforce]`; closes #89
- **`BigQueryLoader`** — load BigQuery table rows or query results via `google-cloud-bigquery`; supports both `table` (full scan) and `query` modes; `pip install synapsekit[bigquery]`; closes #91
- **Subgraph checkpoint scoping** — each subgraph execution gets its own scoped checkpoint ID (`parent::name::step`); failed subgraphs can be resumed independently without restarting the parent graph; `subgraph_node()` gains optional `name` parameter (default `"subgraph"`); new `CompiledGraph.resume_subgraph()` convenience method; works with all checkpointer backends (InMemory, SQLite, JSON, Redis, Postgres); closes #252
- **`VoiceAgent`** — end-to-end voice pipeline (STT → agent → TTS); pluggable STT backends: OpenAI Whisper API (`whisper-1`) and local (`faster-whisper` / `openai-whisper`); pluggable TTS backends: OpenAI TTS API and `pyttsx3`; three I/O modes: `run_file()` (audio file in, audio file out), `run_stream()` (mic → speaker), and `ws_handler()` (WebSocket for web apps); zero-dep energy-based VAD (`EnergyVAD`) to filter silence; `pip install synapsekit[voice]` for OpenAI backends, `pip install synapsekit[voice-local]` for local backends, `pip install synapsekit[voice-stream]` for mic/speaker streaming; closes #592
- **`StateGraph` JSON serialization** — `to_json()` / `from_json()` roundtrip for full graph definitions (nodes, static edges, conditional edges, entry point, checkpointer config); `from_json()` accepts optional `node_factories` and `condition_factories` dicts for custom node/condition reconstruction; closes #597
- **Visual Graph Builder** — `synapsekit graph-builder` CLI command launches a local FastAPI + Mermaid.js web UI; edit graph JSON in sidebar, live-preview the Mermaid diagram, generate Python scaffolding, and download as `.py`; REST endpoints: `POST /api/mermaid`, `POST /api/codegen`, `GET /api/schema`; `pip install synapsekit[graph-ui]`; closes #597
- **Agent Benchmarking Suite** — extensible benchmark registry in `synapsekit.evaluation.benchmarks`; stub implementations for GAIA, SWE-bench, WebArena, and AgentBench with `load_dataset()` + `evaluate()` interface; `synapsekit benchmark list` enumerates available suites; `synapsekit benchmark run <suite> <module:agent_fn> [--split] [--limit]` runs evaluation and prints a leaderboard report; closes #594
- **11 new vector store backends** — `VespaVectorStore` (Vespa HTTP Document/Search API); `RedisVectorStore` (RediSearch vector index); `ElasticsearchVectorStore` (dense_vector knn); `OpenSearchVectorStore` (knn_vector); `SupabaseVectorStore` (pgvector via Supabase JS-compatible REST); `TypesenseVectorStore` (Typesense vector search); `MarqoVectorStore` (Marqo structured index); `ZillizVectorStore` (Zilliz Cloud / Milvus SaaS); `DuckDBVectorStore` (in-process DuckDB + VSS extension); `ClickHouseVectorStore` (ClickHouse `array_cosine_similarity`); `CassandraVectorStore` (DataStax Astra DB via `astrapy` or `cassandra-driver`); all follow the standard `add(texts, metadata)` / `search(query, top_k, metadata_filter)` interface; `pip install synapsekit[vespa|redis|elasticsearch|opensearch|supabase|typesense|marqo|zilliz|duckdb-vector|clickhouse|cassandra]`; closes #117 #118 #119 #120 #122 #123 #124 #126 #128 #131 #132
- **9 new document loaders** — `FirestoreLoader` (Firestore collection query via `google-cloud-firestore`); `ZendeskLoader` (Zendesk tickets via REST API); `IntercomLoader` (Intercom conversations via REST API); `FreshdeskLoader` (Freshdesk tickets via REST API); `HackerNewsLoader` (HN Algolia API, configurable story types and limits); `RedditLoader` (Reddit posts/comments via REST API); `TwitterLoader` (Twitter/X recent search via API v2); `GoogleCalendarLoader` (Google Calendar events via Google API client); `TrelloLoader` (Trello cards across boards/lists via Trello REST API); all async `aload()` via executor; closes #93 #102 #103 #104 #105 #106 #107 #111
- **`RAPTORRetriever`** — recursive abstractive processing: embeds documents, clusters via K-Means, summarises each cluster with an LLM, and builds a multi-level tree of summaries; retrieval walks the tree top-down for broad-to-specific coverage; configurable `n_clusters`, `max_levels`, `summarize_prompt`; `pip install synapsekit[semantic]`; closes #151
- **`AgenticRAGRetriever`** — tool-using retrieval agent: initial retrieval then iteratively decides whether to issue follow-up search queries based on LLM judgment; configurable `max_iterations`; deduplicates results across iterations; closes #161
- **`DocumentAugmentationRetriever`** — augment both queries (HyDE-style) and retrieved documents before re-ranking; `expand_queries` generates N hypothetical queries via LLM; `expand_documents` appends LLM-generated context to each result; fully configurable prompts; closes #162
- **`LateChunkingRetriever`** — late-chunking strategy: embed the full document context first, then split; captures long-range context that early-chunking misses; configurable `chunk_size` and `chunk_overlap`; wraps any base retriever; closes #164
- **`ReplicateLLM`** — LLM provider backed by Replicate's model hosting platform; lazy `replicate` client; configurable `version`; streams tokens via `replicate.stream()`; `pip install synapsekit[replicate]`; closes #173
- **`TimedResumeGraph`** — time-based auto-resume for interrupted graphs; wraps any `CompiledGraph`; persists pending state with a target resume timestamp; background asyncio task wakes up at the right time and resumes execution; configurable `resume_delay`, `max_retries`, and `retry_backoff`; closes #245
- **`SwarmAgent`** — dynamic multi-agent swarm: spawns specialist sub-agents based on task complexity analysis; coordinator LLM decomposes tasks into subtasks and routes each to the best-fit agent from the pool; results aggregated into a final synthesised response; async parallel subtask execution via `asyncio.gather`; closes #581
- **`EventTrigger`** — webhook-triggered agent execution; registers an HTTP endpoint (via FastAPI) that fires an agent on matching events; configurable `event_type` filter, payload schema validation, and async handler; closes #582
- **`StreamTrigger`** — event-driven agent activation from Kafka topics and Redis Streams; configurable `consumer_group`, `batch_size`, `poll_interval`; auto-commit on successful agent run; closes #584
- **`synapsekit ui` command** — local observability dashboard served via FastAPI + Uvicorn; displays live trace logs, token usage, latency histograms, and quality metrics from `TokenTracer`; REST API at `/api/traces`, `/api/summary`; `pip install synapsekit[graph-ui]`; closes #585
- **`PluginRegistry` + `BasePlugin`** — first-party plugin system in `synapsekit.plugins`; `BasePlugin` ABC with `name`, `version`, `description` class vars and `on_load()` / `on_unload()` async lifecycle hooks; `PluginRegistry` central registry with `register()`, `await load(name)`, `await unload(name)`, `list_plugins()`, `get(name)`; `synapsekit plugin list|load|unload` CLI commands; closes #586

### Fixed

- **`CronTrigger.prompt`** — renamed `input` parameter to `prompt`; `input` was shadowing the Python builtin `input()` function making it impossible to call the builtin inside trigger callbacks
- **`SimpleAgent.api_key` default** — `api_key` was a required positional-style kwarg; local models (Ollama, LMStudio, LlamaCpp) that need no API key now work without passing an empty string
- **`TokenTracer` zero-data quality fields** — `tracer.summary()` returned `avg_faithfulness=0.0` and `avg_relevancy=0.0` when no quality records existed, falsely indicating zero scores; both fields now return `None` when no quality data has been recorded
- **`SnowflakeLoader` LIMIT applied server-side** — `limit` was previously applied in Python after `cursor.fetchall()` which pulled all rows from Snowflake before truncating; `LIMIT N` is now appended to the SQL query so Snowflake enforces it at query execution time
- **`VoiceAgent.ws_handler` silent error swallow** — bare `except Exception: pass` replaced with `warnings.warn` so WebSocket disconnects and errors are visible instead of silently dropped
- **`VoiceAgent.run_stream` numpy import in hot loop** — `import numpy as np` was inside the per-utterance processing block; hoisted to the top of `run_stream` alongside other deferred imports to avoid repeated module-lookup overhead
- **`BaseBenchmark.name` forced instantiation** — `name` was an `@property` requiring `cls()` instantiation just to display it in `synapsekit benchmark list`; converted to a `ClassVar[str]` on all four benchmark classes so `cls.name` works without side effects
- **Benchmark stub TODO clarity** — stray `# success += 1` comment in GAIA, SWE-bench, WebArena, and AgentBench `evaluate()` methods replaced with explicit `# TODO:` comments describing what correctness check must be implemented per benchmark
- **Mermaid CDN version unpinned** — `graph_builder.py` imported Mermaid from `mermaid@10` (floating); pinned to `mermaid@10.9.3` to prevent silent breakage from upstream changes
- **`graph-ui` extra missing from pyproject.toml** — `synapsekit graph-builder` depended on `fastapi` and `uvicorn` but these were not declared as an installable extra; added `graph-ui = ["fastapi>=0.110", "uvicorn[standard]>=0.29"]`

### Performance

- **Semantic cache O(n) → O(1) lookup** — `SemanticCache` now L2-normalises vectors on insertion and stacks them into a matrix that is rebuilt lazily; lookup is a single batched `matrix @ query_vec` BLAS call instead of a Python for-loop over 256 individual dot products; closes #568
- **Vector store O(1) amortised inserts** — `InMemoryVectorStore.add()` queues batches in a pending list and consolidates via one `np.vstack` at search time, eliminating the previous `np.concatenate` on every insert that caused O(n²) total memory copies; closes #569
- **Vector store O(result) metadata filtering** — inverted index (`field → value → set[doc_idx]`) built and maintained on every `add()`; metadata filter queries now intersect small sets instead of scanning all N documents linearly; closes #574
- **MMR precomputed similarity matrix** — `search_mmr()` computes the full `(fetch_k × fetch_k)` pairwise similarity matrix with one BLAS call before the greedy loop, replacing O(top_k × fetch_k × selected) Python-level dot-product recomputation; closes #572
- **Async DNS in web scraper** — `socket.gethostbyname()` in the SSRF guard now runs in the thread-pool executor (`loop.run_in_executor`) so it never blocks the asyncio event loop; closes #570
- **Persistent HTTP session in `HTTPRequestTool`** — single `aiohttp.ClientSession` created lazily and reused across all calls on the same tool instance; eliminates TCP + TLS handshake overhead on every request; `aclose()` and async context manager protocol added; closes #571
- **Rate limiter sleep moved outside lock** — `TokenBucketRateLimiter.acquire()` releases the `asyncio.Lock` before calling `asyncio.sleep`, so multiple concurrent callers each wait independently instead of being serialised behind a single sleeper; closes #573
- **Ensemble retriever parallel fan-out** — `EnsembleRetriever.retrieve()` uses `asyncio.gather` to query all retrievers concurrently; total latency is now bounded by the slowest retriever rather than the sum; closes #576
- **Cache key generation overhead removed** — dropped redundant `sort_keys=True` from `json.dumps` in `AsyncLRUCache.make_key`; Python 3.7+ dict insertion order is already stable so sorting was redundant O(k log k) work on every cache lookup; closes #577
- **SQLite cache guaranteed close** — `SQLiteLLMCache` now implements `__enter__` / `__exit__` / `__del__` so the connection is guaranteed to close on all exit paths; `close()` is idempotent; closes #578
- **Evaluation metrics parallel execution** — `EvaluationPipeline.evaluate()` runs all metrics concurrently via `asyncio.gather`; `evaluate_batch()` also parallelises samples with a configurable `asyncio.Semaphore` (default `concurrency=10`); closes #575
- **Sitemap BFS queue O(1) dequeue** — `SitemapLoader._collect_urls()` BFS queue switched from `list.pop(0)` (O(n)) to `collections.deque.popleft()` (O(1)); closes #579

---

## [1.5.6] — 2026-04-16

### Added

- **`GPT4AllLLM`** — local model provider via GPT4All's Python bindings; loads GGUF models locally with no API key; streaming via callback shim (blocking `generate()` wrapped in `run_in_executor` for async safety); `pip install synapsekit[gpt4all]`; closes #548
- **`VLLMLlm`** — high-throughput local/self-hosted inference via vLLM's OpenAI-compatible API; reuses `AsyncOpenAI` client with custom `base_url`; supports streaming, tool calling, and temperature/max-token overrides; `pip install synapsekit[vllm]`; closes #547
- **`SQLiteVecStore`** — zero-infra vector store backed by `sqlite-vec`; stores and retrieves embeddings in a local SQLite file; drop-in replacement for `InMemoryVectorStore` when persistence across sessions is needed; `pip install synapsekit[sqlite-vec]`; closes #545
- **`ParquetLoader`** — load Parquet files as Documents; configurable `text_column` for body text; remaining columns become metadata; supports local files and URLs via `pandas.read_parquet`; one Document per row; async `aload()` via executor; `pip install synapsekit[parquet]`; closes #546
- **`RedisLoader`** — load key/value pairs from a Redis database as Documents; supports string, hash, and JSON value types (via `json.loads`); key pattern filtering via `scan_iter`; metadata includes key and value type; `pip install synapsekit[redis]`; closes #544
- **`ElasticsearchLoader`** — load documents from an Elasticsearch index as Documents; supports both `search` (query DSL) and full `scan` modes; configurable `text_field` and `metadata_fields`; async `aload()` via executor; `pip install synapsekit[elasticsearch]`; closes #543
- **`DynamoDBLoader`** — load items from an AWS DynamoDB table as Documents; supports `scan` (full table) and `query` (key condition) modes with automatic pagination; `text_fields` concatenated as body text; remaining fields become metadata; deserialises typed DynamoDB attribute values; `pip install synapsekit[dynamodb]`; closes #540
- **Production-grade test suite** — four new test categories added; `tests/preflight/` (35 smoke tests: version, imports, async contracts), `tests/e2e/` (RAG, Graph, Agent full-pipeline tests with mocked LLMs), `tests/behavioral/` (Memory, Tracer, Pipeline edge-case tests), `tests/api/` (FastAPI endpoint tests, MCP server handler tests); all tests run with zero API calls and zero network dependencies

### Fixed

- **Stream disconnect race condition** — fixed a race condition where a client disconnect during streaming would raise an unhandled exception instead of cleanly terminating the generator; closes #554
- **Summary buffer memory corruption** — fixed a bug where the summary buffer was being mutated before the summarisation LLM call completed, causing stale or partial summaries under concurrent usage; closes #552
- **`__version__` stale in `__init__.py`**  — `src/synapsekit/__init__.py` was hardcoded to `1.3.0` while `pyproject.toml` was `1.5.5`; now reads dynamically or is kept in sync; caught by the new preflight `test_version_matches_pyproject` test

---

## [1.5.5] — 2026-04-13

### Added

- **`S3Loader`** — load files from Amazon S3 buckets into Documents; supports text, binary fallback, and rich file extraction (PDF, DOCX, XLSX, PPTX, CSV, JSON, HTML) via existing loaders; prefix filtering, extension filtering, `max_files` limit; credential chain (explicit keys, session tokens, or ambient IAM role); async `aload()` via executor; `pip install synapsekit[s3]`; closes #522
- **`AzureBlobLoader`** — load blobs from Azure Blob Storage containers; supports both connection-string and account URL + credential auth; same extraction chain as S3Loader; prefix filtering, `max_files`; `pip install synapsekit[azure]`; closes #520
- **`MongoDBLoader`** — load documents from a MongoDB collection as Documents; configurable `text_fields` and `metadata_fields`; optional `query_filter`; projection builder fetches only requested fields; defensive copy of filter dict; sync + async; `pip install synapsekit[mongodb]`; closes #519
- **`DropboxLoader`** — load files from a Dropbox folder; supports 20+ text/code extensions; pagination via cursor; `limit` stops fetching early; download-error skipping; `pip install synapsekit[dropbox]`; closes #517
- **`EvalDataset` / `EvalRecord`** — filterable, exportable collection of eval result records; `filter_score(min_score, max_score)` narrows to weak/strong cases; `export()` writes fine-tuning datasets in OpenAI, Anthropic, Together, JSONL, and DPO pair formats; `from_snapshot()` loads from existing EvalCI snapshots
- **`FineTuner`** — orchestrates fine-tuning jobs against OpenAI and Together AI; injectable adapter pattern for extensibility; `submit()`, `status()`, `wait()` (polls until terminal state with configurable timeout/interval); `FineTuneJob` dataclass tracks id, provider, status, model_id, error
- **`@eval_case(capture_io=True)`** — opt-in capture of `input`, `output`, and `ideal` fields in eval case results; required for `EvalDataset.export()`
- **`synapsekit eval` CLI** — `report <snapshot>` summarises scores and weak cases; `export <snapshot> --format openai --output data.jsonl` writes fine-tune dataset; `compare <baseline> <current>` runs regression comparison
- **`synapsekit finetune` CLI** — `submit <dataset> --provider openai --base-model gpt-4o-mini`; `status <job_id>`; `wait <job_id>` blocks until completion
- **Subgraph Checkpoint Scoping** — each subgraph execution now gets its own checkpoint scope via a scoped `graph_id` (`parent::name::step`); subgraph state is checkpointed independently so failed subgraphs can be resumed without restarting the parent; `subgraph_node()` gains optional `name` parameter; `CompiledGraph.resume_subgraph()` convenience method; works with all existing checkpointer backends (InMemory, SQLite, JSON, Redis, Postgres); 11 new tests
- **Recursive Subgraph Support** — allow a `StateGraph` to be passed to `subgraph_node()`, enabling self-referential / recursive workflows; implements a `max_recursion_depth` guard (default 10) to prevent infinite loops; tracks depth via internal `__recursion_depth__` state key; adds `RecursionDepthError` to handle limit breaches; lazy compilation supports definition-time self-referencing.
- **Discord community link** — added Discord server link to README community section.
- **`LMStudioLLM`** — local model provider via LM Studio's OpenAI-compatible API; connects to a running LM Studio server (default `http://localhost:1234/v1`); supports streaming, tool calling, and custom `base_url` via constructor kwarg; no API key required; `pip install synapsekit[lmstudio]`; closes #176
- **`MCPServer` SSE transport + package refactor** — `MCPServer` now lives in `synapsekit.mcp.server` package; adds `run_sse(host, port, api_key)` for HTTP/SSE MCP serving with optional Bearer auth; backwards-compatible with existing `MCPServer(tools=[...])`, `MCPServer(rag)`, and `MCPServer(agent)` usage
- **`LaTeXLoader`** — load `.tex` files as plain text; strips commands, environments, inline/display math, and comments via regex; captures section/subsection titles into metadata; no external deps required
- **`TSVLoader`** — load tab-separated files one Document per row; configurable `text_column` to extract a specific column as text; remaining columns become metadata; skips empty rows; async `aload()`
- **`RTFLoader`** — load RTF files as plain text via `striprtf`; handles malformed RTF gracefully; `pip install synapsekit[rtf]`
- **`EPUBLoader`** — load EPUB files chapter-by-chapter; extracts title, author, and chapter name into metadata; strips HTML tags safely; `pip install synapsekit[epub]`
- **`ConfigLoader`** — load `.env`, `.ini`, `.cfg`, `.toml`, and environment-specific dotfiles (`.env.local`, `.env.staging`, `.env.production`) into Documents; redacts sensitive keys (password, secret, token, api_key, auth) automatically; one Document per INI section; Python 3.11+ uses stdlib `tomllib`, falls back to `tomli` on older versions
- **`OneDriveLoader`** — load files from OneDrive and SharePoint via Microsoft Graph API; folder traversal with optional recursion; extension filtering; `max_files` cap; extracts PDF, DOCX, XLSX, PPTX, CSV, JSON, HTML via existing loaders; async `aload()`; uses stdlib HTTP (no external SDK required)

### Fixed

- **`DropboxLoader` SDK compatibility** — original implementation called `.get()` on Dropbox SDK entry objects (`FileMetadata`, `FolderMetadata`), which are Stone-generated Python classes, not dicts; fixed with `_normalise_entry()` static method that converts SDK objects to canonical dicts via attribute access while passing test-mock dicts through unchanged
- **`LMStudioLLM` `base_url`** — `LLMConfig` has no `base_url` field; passing it via `LLMConfig(base_url=...)` would raise `TypeError` before `LMStudioLLM.__init__` ran. Fixed by adding `base_url: str | None = None` as a keyword argument to `LMStudioLLM.__init__` directly (mirrors the `XaiLLM` / `NovitaLLM` pattern). Custom server usage: `LMStudioLLM(config, base_url="http://192.168.1.10:1234/v1")`
- **`LMStudioLLM` stream stability** — removed `stream_options={"include_usage": True}` which caused API errors on older LM Studio builds; usage tracking now reads `chunk.usage` defensively via `getattr` so it still captures tokens when the server returns them
- **`ConfigLoader` rejects `.env.local` / `.env.staging`** — `os.path.splitext(".env.local")` returns `('.env', '.local')` making `ext = '.local'` which fell through to `ValueError: Unsupported config file type`. Fixed by detecting any file whose basename starts with `.env` and treating it as the env format regardless of secondary extension
- **`RTFLoader` default encoding** — changed default from `"utf-8"` to `"latin-1"` (Windows-1252 superset) since real-world RTF files from Office/WordPad are almost universally Windows-encoded, not UTF-8

---

## [1.5.3] — 2026-04-11

### Added

- **`TeamsLoader`** — load messages from Microsoft Teams channels via the Microsoft Graph API; automatic pagination via `@odata.nextLink`; HTML-to-plain-text conversion (strips tags, decodes entities); exponential backoff retry for 429 and 5xx responses; graceful handling of missing author/timestamp/body fields; sync `load()` and async `aload()`; `pip install synapsekit[teams]`; closes #51
- **`CodeInterpreterTool`** — execute Python code in an isolated subprocess and capture stdout, stderr, generated files, matplotlib plot artifacts, and pandas dataframe reprs; configurable timeout (default 5s) and memory limit (default 256 MB, enforced via `RLIMIT_AS` on Linux); workspace isolation via `tempfile.TemporaryDirectory`; structured JSON output; closes #216

### Fixed

- **`ShellTool` Windows compatibility** — use `asyncio.create_subprocess_shell()` on Windows so shell builtins (`echo`, `dir`, etc.) work correctly; keep `create_subprocess_exec()` on Unix; closes #502
- **pytest Windows warnings** — suppress harmless `PytestUnraisableExceptionWarning` from asyncio proactor transport GC cleanup on Windows

---

## [1.5.2] — 2026-04-10

### Added

- **`JSONSplitter`** — JSON-aware chunking; splits arrays by element and objects by top-level key; item-level overlap preserves valid JSON structure (character-level overlap would produce invalid JSON fragments); closes #501
- **EvalCI GitHub Action** — live on [GitHub Marketplace](https://github.com/marketplace/actions/evalci-by-synapsekit); LLM quality gates on every PR, zero infrastructure, 2-minute setup

### Fixed

- **Async `@eval_case` not awaited** — decorated async functions now correctly preserve `inspect.iscoroutinefunction()` identity; the CLI runner was skipping `asyncio.run()` for async cases, passing a raw coroutine to `float()` and raising `TypeError`; regression tests added

---

## [1.5.1] — 2026-04-09

### Added

- **WeatherTool tests** — full test suite covering current weather, forecast, missing API key, empty results, network errors, schema validation; closes #383
- **`GitLoader`** — load files from any Git repository (local path or remote URL) at a specific revision; glob pattern filtering; metadata includes path, commit hash, author, date; `pip install synapsekit[git]`
- **`GoogleSheetsLoader`** — load rows from a Google Sheets spreadsheet as Documents; service account auth via credentials file; auto-detects first sheet if none specified; header-based row-to-text formatting; `pip install synapsekit[gsheets]`
- **`JiraLoader`** — load Jira issues via JQL queries; full Atlassian Document Format (ADF) parsing; pagination; rate-limit retry; async `aload()` via httpx; optional `limit`; `pip install synapsekit[jira]`
- **`SupabaseLoader`** — load rows from a Supabase table as Documents; configurable text/metadata columns; env var auth (`SUPABASE_URL`, `SUPABASE_KEY`); `pip install synapsekit[supabase]`

### Security

- **SQL injection** — `SQLSchemaInspectionTool` now validates table names against `^[A-Za-z0-9_]+$` before interpolating into `PRAGMA table_info()`; closes #494
- **Shell injection** — `ShellTool` switched from `create_subprocess_shell` to `create_subprocess_exec` with `shlex.split()`; allowlist now checked against the actual binary (`argv[0]`) instead of a whitespace split; closes #495
- **Path traversal** — `FileReadTool` and `FileWriteTool` now accept an optional `base_dir`; all paths are resolved with `Path.resolve()` and checked to be within the sandbox before I/O; closes #496
- **TOCTOU in VideoLoader** — replaced `tempfile.mktemp()` with `tempfile.NamedTemporaryFile(delete=False)` to eliminate the race window; closes #497
- **SSRF** — `WebLoader` and `WebScraperTool` now validate URL scheme (must be `http`/`https`) and block requests to private/internal IP ranges (RFC 1918, loopback, link-local, IPv6 ULA); closes #498
- **ReDoS** — `WebScraperTool` limits CSS selector length to 200 characters; closes #498

---

## [1.5.0] — 2026-04-07

### Added

- **RSSLoader** — load articles from RSS/Atom feeds as Documents; content/summary fallback; metadata includes title, published, link, author; async `aload()`; `pip install synapsekit[rss]`
- **SentenceTextSplitter** — split text into chunks by grouping complete sentences; `chunk_size` and `chunk_overlap` in sentences; regex-based sentence boundary detection
- **CodeSplitter** — split source code using language-aware separators; supports Python, JavaScript, TypeScript, Go, Rust, Java, C++; preserves logical structures (classes, functions); falls back to recursive character splitting
- **ConfluenceLoader** — load pages from Atlassian Confluence as Documents; supports single page by `page_id` or full space by `space_key`; automatic pagination; converts Confluence storage format to plain text via BeautifulSoup; rich metadata (title, author, version, URL); retry with exponential back-off for rate limits; sync `load()` and async `aload()`; `pip install synapsekit[confluence]`
- **SentenceWindowSplitter** — splits text into one chunk per sentence, each padded with up to `window_size` surrounding sentences for context; custom `split_with_metadata()` adds `target_sentence` to chunk metadata; useful for retrieval systems that embed with context but index by target sentence
- **XaiLLM** — xAI Grok LLM provider; OpenAI-compatible API; supports grok-beta, grok-2, grok-2-mini; streaming and tool calling; auto-detected from model name; `pip install synapsekit[openai]`
- **WeatherTool** — get current weather and short-term forecasts via OpenWeatherMap; actions: current, forecast (1-5 day); async-safe with run_in_executor; auth via OPENWEATHERMAP_API_KEY
- **WriterLLM** — Writer (Palmyra) LLM provider; OpenAI-compatible API; supports palmyra-x-004, palmyra-x-003-instruct, palmyra-med, palmyra-fin; streaming and tool calling; `pip install synapsekit[openai]`
- **NovitaLLM** — NovitaAI LLM provider; OpenAI-compatible API; supports Llama, Mistral, Qwen, and other open models; streaming and tool calling; `pip install synapsekit[openai]`
- **StripeTool** — read-only Stripe data lookup: get_customer, list_invoices, get_charge, list_products; stdlib urllib only; auth via STRIPE_API_KEY; async-safe with run_in_executor
- **HTMLTextSplitter** — split HTML documents on block-level tags (h1-h6, p, div, section, article, li, blockquote, pre); strips tags to plain text; falls back to RecursiveCharacterTextSplitter for long sections; stdlib html.parser only
- **TwilioTool** — send SMS and WhatsApp messages via the Twilio REST API; stdlib `urllib` only, no extra deps; auth via constructor args or `TWILIO_ACCOUNT_SID` / `TWILIO_AUTH_TOKEN` / `TWILIO_FROM_NUMBER` env vars; automatic `whatsapp:` prefix handling for both sender and recipient; security warning logged on instantiation; closes #386
- **LinearTool** — manage Linear issues via the Linear GraphQL API; actions: list_issues, get_issue, create_issue, update_issue; stdlib urllib only, no extra deps; auth via constructor arg or `LINEAR_API_KEY` env var; closes #387
- **GCSLoader** — load files from Google Cloud Storage buckets as Documents; supports service account auth (file path or dict) or default credentials; prefix filtering, max_files limit, binary file handling; sync `load()` and async `aload()`; `pip install synapsekit[gcs]`; closes #56
- **SQLLoader** — load rows from any SQLAlchemy-supported database (PostgreSQL, MySQL, SQLite, etc.) as Documents; configurable text/metadata columns; full SQL query support; sync `load()` and async `aload()`; `pip install synapsekit[sql]`; closes #58
- **GitHubLoader** — load README, issues, pull requests, or repository files from GitHub via the REST API; retry with exponential back-off for rate limits and 5xx; optional token auth for higher rate limits; path filtering and limit for files; uses existing `httpx` dep; sync `load_sync()` and async `load()`; closes #48
- **NewsTool** — fetch top headlines and search articles via NewsAPI; actions: get_headlines, search; stdlib urllib only; auth via constructor arg or NEWS_API_KEY env var; closes #384

---

## [1.4.8] — 2026-04-03

### Added

- **WikipediaLoader** — load Wikipedia articles as Documents via `wikipedia-api`; pipe-delimited multi-title support; async `aload()`; `pip install synapsekit[wikipedia]`
- **ArXivLoader** — search arXiv and load papers as Documents (downloads PDFs); uses arxiv v2 `Client` API; async `aload()`; `pip install synapsekit[arxiv,pdf]`
- **EmailLoader** — load emails from IMAP mailboxes (Gmail, Outlook, etc.) as Documents; stdlib-only, no extra deps; configurable folder and IMAP search; async `aload()`
- **ColBERTRetriever** — late-interaction ColBERT retrieval via RAGatouille; `add()`, `retrieve()`, `retrieve_with_scores()`; lazy-loads ragatouille on first use; `pip install synapsekit[colbert]`
- 50 new tests (1500 total)

---

## [1.4.7] — 2026-04-02

### Added

- **SlackLoader** — load messages from Slack channels via Bot API; sync `load()` and async `aload()`; configurable `limit`; per-message metadata; `pip install synapsekit[slack]`
- **NotionLoader** — load pages or full databases from Notion via the Notion API; sync `load()` and async `aload()`; configurable retry/timeout; `pip install synapsekit[notion]`
- **NotionTool** — agent tool for Notion: `search`, `get_page`, `create_page`, `append_block`; built-in retry with exponential back-off; `pip install synapsekit[notion]`
- **Subgraph error handling** — `subgraph_node()` gains `on_error` (`"raise"` / `"retry"` / `"fallback"` / `"skip"`), `max_retries`, and `fallback` parameters
- 17 new tests (1467 total)

---

## [1.4.6] — 2026-04-01

### Added

- **Subgraph error handling** — `subgraph_node()` gains four keyword-only parameters: `on_error` (`"raise"` / `"retry"` / `"fallback"` / `"skip"`), `max_retries` (default 3), and `fallback` (a secondary `CompiledGraph`). On any handled failure the parent state receives `"__subgraph_error__"` with `"type"`, `"message"`, and `"attempts"` keys. Backward-compatible: default behaviour (`on_error="raise"`) is unchanged. (#378)
- 17 new tests (1450 total)

---

## [1.4.5] — 2026-03-31

### Added

- **WeaviateVectorStore** — Weaviate v4 client; lazy collection creation; cosine vector search via `query.near_vector`; metadata filtering; `pip install synapsekit[weaviate]`
- **PGVectorStore** — PostgreSQL + pgvector; async psycopg3; cosine / L2 / inner-product distance; SQL-injection-safe via `psycopg.sql.Identifier`; metadata JSONB; `pip install synapsekit[pgvector]`
- **MilvusVectorStore** — IVF_FLAT and HNSW index types; `MilvusIndexType` enum; metadata filtering via Milvus expressions; Zilliz Cloud support; `pip install synapsekit[milvus]`
- **LanceDBVectorStore** — embedded, serverless; local and cloud (S3/GCS) storage; automatic FTS index; metadata filtering; `pip install synapsekit[lancedb]`

All four backends follow the existing lazy-import `_BACKENDS` pattern and are included in `synapsekit[all]`.

- 0 new tests (1433 total)

---

## [1.4.4] — 2026-03-30

### Added

- **SambaNova provider** — fast inference on Meta Llama, Qwen, and other open models via SambaNova Cloud's OpenAI-compatible API; requires `pip install synapsekit[openai]`; always set `provider="sambanova"`
- **GoogleDriveLoader** — load files and folders from Google Drive via service-account credentials; supports Google Docs (text export), Sheets (CSV export), PDFs, and text files; `pip install synapsekit[gdrive]`
- **`split_with_metadata()`** — new method on `BaseSplitter`; returns `list[dict]` with `text` and `metadata` keys; automatically injects `chunk_index`; all splitters inherit it

### Fixed

- `asyncio.get_event_loop()` → `asyncio.get_running_loop()` in `GoogleDriveLoader` (deprecated in Python 3.10+)
- `build()` in `GoogleDriveLoader.aload()` wrapped in executor (was blocking the event loop)
- Failed file downloads in `GoogleDriveLoader._load_folder` now log a warning instead of silently skipping

- 49 new tests (1452 total)

---

## [1.4.3] — 2026-03-29

### Added

- **XMLLoader** — load XML files via stdlib `xml.etree.ElementTree`; optional `tags` filter; no new dependencies
- **DiscordLoader** — load messages from Discord channels via bot token; `before_message_id` / `after_message_id` pagination; rich metadata (author, timestamp, attachments); `pip install synapsekit[discord]`
- **PythonREPLTool timeout** — `timeout: float = 5.0` parameter; Unix uses `signal.SIGALRM`, Windows uses `multiprocessing.Process`; security warning logged on instantiation

### Improved

- **Mermaid conditional edges** — render as dashed arrows (`-.->`) to distinguish from deterministic edges; branch labels prefixed with condition function name (e.g. `route:approve`)
- **SQLiteCheckpointer** — supports `async with` for automatic connection cleanup
- **Windows compatibility** — `audio/x-wav` MIME normalised to `audio/wav`; shell timeout test uses portable Python sleep; graph tracer uses `time.perf_counter()` for sub-millisecond resolution

- 35 new tests (1403 total)

---

## [1.4.2] — 2026-03-28

### Added

- **HuggingFaceLLM** — Hugging Face Inference API via `AsyncInferenceClient`; supports serverless API (model ID) and Dedicated Inference Endpoints (URL); streaming and generate
- **DynamoDBCacheBackend** — serverless LLM response caching on AWS DynamoDB; configurable partition key, TTL via `ttl_seconds`, custom region and endpoint
- **MemcachedCacheBackend** — high-throughput distributed LLM caching via `aiomcache`; TTL via `exptime`; async-native
- **GoogleSearchTool** — Google web search via SerpAPI (`google-search-results`); graceful handling of missing keys, empty results, and API errors
- **Graph versioning and checkpoint migration** — `StateGraph(version="1", migrations={...})`; `CompiledGraph.resume()` now applies migration chains when loading older checkpoint versions; supports direct and chained `(next_version, state_dict)` migrations; missing paths raise `GraphRuntimeError`
- 11 new tests (1368 total)

### Improved

- **SQLQueryTool** — added `params` dict for parameterized queries (eliminates string interpolation); `max_rows` cap; fixed variable name shadowing

---

## [1.4.1] — 2026-03-27

### Added

- **MinimaxLLM provider** — `MinimaxLLM` for Minimax API with SSE streaming; requires `group_id`; auto-detected from `minimax-*` model prefix
- **AlephAlphaLLM provider** — `AlephAlphaLLM` for Aleph Alpha Luminous and Pharia models; httpx-based streaming; auto-detected from `luminous-*`/`pharia-*` prefixes
- **YAMLLoader** — load YAML files (list-of-objects or single-object) into `Document` list; configurable `text_key` and `metadata_keys`; uses `yaml.safe_load()`
- **BingSearchTool** — web search via Bing Web Search API v7; auth via `Ocp-Apim-Subscription-Key`; supports `query`, `count` (capped at 50), `market`
- **WolframAlphaTool** — computational queries via Wolfram Alpha API; short-answer endpoint; thread-executor async wrapper
- **Usage examples** — `examples/` directory with 5 runnable scripts: RAG quickstart, agent with tools, graph workflow, multi-provider, caching & retries
- 30 new tests (1357 total)

### Fixed

- Added missing return type annotations to `_loader_for()` in `loaders/directory.py` and `__getattr__()` in `loaders/__init__.py`

---

## [1.4.0] — 2026-03-25

### Added

- **AI21 Labs provider** — `AI21LLM` for Jamba models (`jamba-1.5-mini`, `jamba-1.5-large`) with 256K context; streaming and native function calling; auto-detected from `jamba-*` model prefix
- **Databricks provider** — `DatabricksLLM` for Databricks Foundation Model APIs (DBRX, Llama 3.1, Mixtral); OpenAI-compatible endpoint; resolves workspace URL from `DATABRICKS_HOST`; auto-detected from `dbrx-*`/`databricks-*`
- **Baidu ERNIE provider** — `ErnieLLM` for ERNIE Bot (`ernie-4.0`, `ernie-3.5`, `ernie-speed`, `ernie-lite`, `ernie-tiny-8k`); native function calling; auto-detected from `ernie-*`
- **llama.cpp provider** — `LlamaCppLLM` for local GGUF models with no API key; queue+thread true streaming; GPU offload via `n_gpu_layers`; auto-detected from `llamacpp` provider string
- **ImageAnalysisTool** — analyze images with any multimodal LLM; accepts local file paths or public URLs; supports OpenAI and Anthropic message formats
- **TextToSpeechTool** — convert text to speech audio via OpenAI TTS; supports `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer` voices and mp3/wav/flac/aac formats
- **SpeechToTextTool** — transcribe audio files via OpenAI Whisper API or local Whisper model; delegates to `AudioLoader`
- **APIBuilderTool** — build and execute API calls from OpenAPI specs or natural-language intent; LLM-assisted operation selection; supports inline specs, spec URLs, and explicit path/method
- **GoogleCalendarTool** — create, list, and delete Google Calendar events via Google Calendar API v3 with Application Default Credentials
- **AWSLambdaTool** — invoke AWS Lambda functions with RequestResponse/Event/DryRun invocation types; standard boto3 credential resolution
- 222 new tests (1327 total)

### Changed

- `RAG` facade auto-detection extended with `moonshot-*`, `glm-*`, `jamba-*`, `@cf/*`, `@hf/*`, `dbrx-*`/`databricks-*`, `ernie-*` model prefixes

---

## [1.0.0] — 2026-03-18

### Added

- **Multimodal support** — `ImageContent` (from_file, from_url, from_base64), `AudioContent`, `MultimodalMessage` with OpenAI/Anthropic format conversion
- **Image loader** — `ImageLoader` with sync/async loading and optional vision LLM description
- **API stability markers** — `@public_api`, `@experimental` (FutureWarning), `@deprecated(reason, alternative)` (DeprecationWarning)
- 42 new tests (1011 total)

---

## [0.9.0] — 2026-03-18

### Added

- **A2A protocol** — `A2AClient`, `A2AServer`, `AgentCard` for Google Agent-to-Agent protocol; `A2ATask`, `A2AMessage`, `TaskState` types
- **Agent guardrails** — `ContentFilter` (blocked patterns/words/max length), `PIIDetector` (email, phone, SSN, credit card, IP), `TopicRestrictor`, `Guardrails` (composite checker)
- **Distributed tracing** — `DistributedTracer` and `TraceSpan` with parent-child relationships, events, and timing
- 64 new tests (1008 total)

---

## [0.8.0] — 2026-03-18

### Added

- **Evaluation metrics** — `FaithfulnessMetric` (claim verification), `RelevancyMetric` (document relevance), `GroundednessMetric` (answer grounding score 0-1)
- **Evaluation pipeline** — `EvaluationPipeline` runs multiple metrics, `EvaluationResult` with mean_score aggregation
- **OpenTelemetry tracing** — `OTelExporter` with optional OTLP export, `Span` spans, `TracingMiddleware` auto-traces LLM calls
- **Tracing UI** — `TracingUI` renders traces as HTML dashboard, saves to file, or serves via local HTTP server
- 50 new tests (944 total)

---

## [0.7.0] — 2026-03-18

### Added

- **MCP client** — `MCPClient` connects to MCP servers via stdio or SSE transport; `MCPToolAdapter` wraps MCP tools as `BaseTool` instances
- **MCP server** — `MCPServer` exposes SynapseKit tools as MCP-compatible tools via stdio or SSE
- **Supervisor agent** — `SupervisorAgent` delegates tasks to `WorkerAgent` instances via DELEGATE/FINAL protocol
- **Handoff chain** — `HandoffChain` with condition-based `Handoff` transfers between agents, returns `HandoffResult`
- **Crew** — `Crew` for role-based multi-agent teams with `CrewAgent`, `Task`, sequential and parallel execution
- 60 new tests (844 total)

---

## [0.6.9] — 2026-03-18

### Added

- **Slack tool** — `SlackTool` sends messages via Slack webhook URL or Web API bot token (`SLACK_WEBHOOK_URL` / `SLACK_BOT_TOKEN` env vars, stdlib only)
- **Jira tool** — `JiraTool` interacts with Jira REST API v2: search issues (JQL), get issue, create issue, add comment (`JIRA_URL`, `JIRA_EMAIL`, `JIRA_API_TOKEN`, stdlib only)
- **Brave Search tool** — `BraveSearchTool` web search via Brave Search API (`BRAVE_API_KEY`, stdlib only)
- **Approval node** — `approval_node()` factory returns a graph node that gates on human approval, raising `GraphInterrupt` when `state[key]` is falsy; supports dynamic messages via callable
- **Dynamic route node** — `dynamic_route_node()` factory returns a graph node that routes to different compiled subgraphs based on a routing function; supports sync/async routing and input/output mapping
- 52 new tests (795 total)

---

## [0.6.8] — 2026-03-18

### Added

- **Email tool** — `EmailTool` sends emails via SMTP with configurable settings or environment variables (`SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`)
- **GitHub API tool** — `GitHubAPITool` searches repos/issues and fetches details via GitHub REST API (stdlib only, no deps)
- **PubMed search tool** — `PubMedSearchTool` searches biomedical literature on PubMed via NCBI E-utilities (stdlib only)
- **Vector search tool** — `VectorSearchTool` wraps any `Retriever` as an agent tool for knowledge base queries
- **YouTube search tool** — `YouTubeSearchTool` searches YouTube videos with titles, channels, durations, view counts (`pip install synapsekit[youtube]`)
- **Execution trace** — `ExecutionTrace` and `TraceEntry` collect and analyze graph execution events with timing, durations, and human-readable summaries
- **WebSocket streaming** — `ws_stream()` streams graph execution over WebSocket connections (works with Starlette, FastAPI, plain websockets)
- `GraphEvent.to_ws()` — JSON serialization for WebSocket transmission
- 45 new tests (743 total)

---

## [0.6.7] — 2026-03-17

### Changed

- **Python version requirement** — raised minimum from `>=3.9` to `>=3.10`
- Added Python 3.14 classifier

---

## [0.6.6] — 2026-03-16

### Added

- **Perplexity LLM** — `PerplexityLLM` for Perplexity AI with Sonar models, OpenAI-compatible
- **Cerebras LLM** — `CerebrasLLM` for Cerebras ultra-fast inference, OpenAI-compatible
- **Hybrid search retrieval** — `HybridSearchRetriever` combines BM25 + vector similarity via Reciprocal Rank Fusion
- **Self-RAG retrieval** — `SelfRAGRetriever` with self-reflective retrieve-grade-generate-check loop
- **Adaptive RAG retrieval** — `AdaptiveRAGRetriever` classifies query complexity and routes to different retrieval strategies
- **Multi-step retrieval** — `MultiStepRetriever` iterative retrieval-generation with gap identification
- **arXiv search tool** — `ArxivSearchTool` searches arXiv for academic papers (stdlib only)
- **Tavily search tool** — `TavilySearchTool` AI-optimized web search via Tavily API
- **Buffer memory** — `BufferMemory` simplest unbounded message buffer
- **Entity memory** — `EntityMemory` LLM-based entity extraction with running descriptions and eviction
- 56 new tests (698 total)

---

## [0.6.5] — 2026-03-15

### Added

- **Cohere reranker** — `CohereReranker` reranks retrieval results using the Cohere Rerank API
- **Step-back retrieval** — `StepBackRetriever` generates step-back questions for broader context + parallel retrieval
- **FLARE retrieval** — `FLARERetriever` Forward-Looking Active REtrieval with iterative `[SEARCH: ...]` markers
- **DuckDuckGo search tool** — `DuckDuckGoSearchTool` extended search with text and news types
- **PDF reader tool** — `PDFReaderTool` reads and extracts text from PDF files with optional page selection
- **GraphQL tool** — `GraphQLTool` executes GraphQL queries against any endpoint
- **Token buffer memory** — `TokenBufferMemory` token-budget-aware memory that drops oldest messages (no LLM)
- **Redis LLM cache** — `RedisLLMCache` distributed Redis cache backend (`pip install synapsekit[redis]`)
- 55 new tests (642 total)

---

## [0.6.4] — 2026-03-15

### Added

- **Docx loader** — `DocxLoader` for Word documents via `python-docx`
- **Markdown loader** — `MarkdownLoader` with optional YAML frontmatter stripping
- **HyDE retrieval** — `HyDERetriever` Hypothetical Document Embeddings retrieval strategy
- **Shell tool** — `ShellTool` shell command execution with timeout and allowed-commands filter
- **SQL schema inspection tool** — `SQLSchemaInspectionTool` lists tables and describes columns
- **Filesystem LLM cache** — `FilesystemLLMCache` persistent JSON file-based cache backend
- **JSON file checkpointer** — `JSONFileCheckpointer` JSON file-based graph checkpoint persistence
- **TokenTracer COST_TABLE** — added GPT-4.1, o3, o4-mini, Gemini 2.5, DeepSeek-V3/R1, Groq models
- 45 new tests (587 total)

---

## [0.6.3] — 2026-03-14

### Added

- **Typed state with reducers** — `TypedState` and `StateField` for safe parallel state merging in graph workflows; per-field reducers control how concurrent node outputs are combined (closes #253)
- **Parallel subgraph execution** — `fan_out_node()` runs multiple subgraphs concurrently with `asyncio.gather()`, supports per-subgraph input mappings and custom merge functions (closes #248)
- **SSE streaming** — `sse_stream()` streams graph execution as Server-Sent Events for HTTP responses (closes #238)
- **Event callbacks** — `EventHooks` and `GraphEvent` for registering callbacks on node_start, node_complete, wave_start, wave_complete events during graph execution (closes #240)
- **Semantic LLM cache** — `SemanticCache` uses embeddings for similarity-based cache lookup instead of exact match; configurable threshold and maxsize (closes #196)
- **Summarization tool** — `SummarizationTool` summarizes text using an LLM with concise, bullet_points, or detailed styles (closes #223)
- **Sentiment analysis tool** — `SentimentAnalysisTool` analyzes text sentiment (positive/negative/neutral) with confidence and explanation (closes #225)
- **Translation tool** — `TranslationTool` translates text between languages with optional source language specification (closes #224)
- 28 new tests (540 total)

---

## [0.6.2] — 2026-03-13

### Added

- **CRAG (Corrective RAG)** — `CRAGRetriever` grades retrieved documents for relevance using an LLM, rewrites the query and retries when too few are relevant (closes #152)
- **Query Decomposition** — `QueryDecompositionRetriever` breaks complex queries into sub-queries, retrieves for each, and deduplicates results (closes #156)
- **Contextual Compression** — `ContextualCompressionRetriever` compresses retrieved documents to only the relevant excerpts using an LLM (closes #146)
- **Ensemble Retrieval** — `EnsembleRetriever` fuses results from multiple retrievers using weighted Reciprocal Rank Fusion (closes #147)
- **SQLite Conversation Memory** — `SQLiteConversationMemory` persists chat history to SQLite with multi-conversation support and optional sliding window (closes #138)
- **Summary Buffer Memory** — `SummaryBufferMemory` tracks token budget and progressively summarizes older messages when the buffer exceeds the limit (closes #135)
- **Human Input Tool** — `HumanInputTool` pauses agent execution to ask the user a question, supports custom sync/async input functions (closes #228)
- **Wikipedia Tool** — `WikipediaTool` searches and fetches Wikipedia article summaries using the REST API, no extra dependencies (closes #202)
- 30 new tests (512 total)

---

## [0.6.1] — 2026-03-13

### Added

- **Human-in-the-Loop** — `GraphInterrupt` exception pauses graph execution for human review; `InterruptState` holds interrupt details; `resume(updates=...)` applies human edits and continues from checkpoint
- **Subgraphs** — `subgraph_node(compiled_graph, input_mapping, output_mapping)` nests a `CompiledGraph` as a node in a parent graph with key mapping
- **Token-level streaming** — `llm_node(llm, stream=True)` wraps any `BaseLLM` as a graph node; `stream_tokens()` yields `{"type": "token", ...}` events for real-time output
- **Self-Query retrieval** — `SelfQueryRetriever` uses an LLM to decompose natural-language queries into semantic search + metadata filters automatically
- **Parent Document retrieval** — `ParentDocumentRetriever` embeds small chunks for precision search, returns full parent documents for richer context
- **Cross-Encoder reranking** — `CrossEncoderReranker` reranks retrieval results with cross-encoder models for higher accuracy (requires `synapsekit[semantic]`)
- **Hybrid memory** — `HybridMemory` keeps a sliding window of recent messages in full, plus an LLM-generated summary of older messages for token-efficient long conversations
- 30 new tests (482 total)

---

## [0.6.0] — 2026-03-13

### Added

- **Built-in tools** (6 new):
  - `HTTPRequestTool` — GET/POST/PUT/DELETE/PATCH with aiohttp, configurable timeout and max response length
  - `FileWriteTool` — write/append files with auto-mkdir
  - `FileListTool` — list directories with glob patterns, recursive mode
  - `DateTimeTool` — current time, parse, format with timezone support
  - `RegexTool` — findall, match, search, replace, split with flag support
  - `JSONQueryTool` — dot-notation path queries on JSON data
- **LLM providers** (3 new, all OpenAI-compatible):
  - `OpenRouterLLM` — unified API for 200+ models (auto-detected from `/` in model name)
  - `TogetherLLM` — Together AI fast inference
  - `FireworksLLM` — Fireworks AI optimized serving
- **Advanced retrieval** (2 new):
  - `ContextualRetriever` — Anthropic-style contextual retrieval (LLM prepends context before embedding)
  - `SentenceWindowRetriever` — sentence-level embedding with configurable window expansion at retrieval time
- RAG facade auto-detects `openrouter` (model names with `/`), `together`, and `fireworks` providers
- 37 new tests (452 total)

### Changed

- Lazy imports extended for new providers (`OpenRouterLLM`, `TogetherLLM`, `FireworksLLM`)
- `agents/tools/__init__.py` exports 11 built-in tools (was 5)

---

## [0.5.3] — 2026-03-12

### Added

- **Azure OpenAI LLM provider** — `AzureOpenAILLM` for enterprise Azure OpenAI deployments with streaming and function calling (closes #183)
- **Groq LLM provider** — `GroqLLM` for ultra-fast inference with Llama, Mixtral, Gemma models (closes #166)
- **DeepSeek LLM provider** — `DeepSeekLLM` with OpenAI-compatible API, supports function calling (closes #170)
- **SQLite LLM cache** — persistent `cache_backend="sqlite"` option via `LLMConfig(cache=True, cache_backend="sqlite")`, survives process restarts (closes #191)
- **RAG Fusion retrieval** — `RAGFusionRetriever` generates query variations with an LLM and fuses results via Reciprocal Rank Fusion for better recall (closes #158)
- **Excel loader** — `ExcelLoader` for `.xlsx` files, one Document per sheet (closes #63)
- **PowerPoint loader** — `PowerPointLoader` for `.pptx` files, one Document per slide (closes #62)
- RAG facade auto-detects `deepseek` and `groq` providers from model names
- 26 new tests (415 total)

### Changed

- `LLMConfig` gains `cache_backend` (`"memory"` or `"sqlite"`) and `cache_db_path` fields
- Lazy imports extended for new providers (`AzureOpenAILLM`, `GroqLLM`, `DeepSeekLLM`) and loaders (`ExcelLoader`, `PowerPointLoader`)

---

## [0.5.2] — 2026-03-12

### Added

- **`__repr__` methods** — human-readable repr on `StateGraph`, `CompiledGraph`, `RAGPipeline`, `ReActAgent`, `FunctionCallingAgent` (closes #3)
- **Empty document handling** — `RAGPipeline.add()` silently skips empty/whitespace-only text instead of producing empty chunks (closes #20)
- **Retry for `call_with_tools()`** — `LLMConfig(max_retries=N)` now applies to native function-calling, not just `generate()` (closes #22)
- **Cache hit/miss statistics** — `BaseLLM.cache_stats` property returns `{"hits", "misses", "size"}` when caching is enabled (closes #23)
- **MMR retrieval** — `InMemoryVectorStore.search_mmr()` and `Retriever.retrieve_mmr()` for diversity-aware retrieval (closes #30)
- **Rate limiting** — `LLMConfig(requests_per_minute=N)` adds token-bucket rate limiting to all LLM calls (closes #35)
- **Structured output with retry** — `generate_structured(llm, prompt, schema=MyModel)` parses LLM output into Pydantic models, retrying on parse failure (closes #43)
- 29 new tests (389 total)

### Changed

- LLM providers now override `_call_with_tools_impl()` instead of `call_with_tools()` (base class handles retry + rate limiting)
- `LLMConfig` gains `requests_per_minute` field (default `None` — off)

---

## [0.5.1] — 2026-03-12

### Added

- **`@tool` decorator** — create agent tools from plain functions with `@tool(name="...", description="...")`; auto-generates JSON Schema from type hints, supports sync and async functions
- **Metadata filtering** — `VectorStore.search(metadata_filter={"key": "value"})` filters results by metadata before ranking; implemented in `InMemoryVectorStore`, signature updated in all backends
- **Vector store lazy exports** — `ChromaVectorStore`, `FAISSVectorStore`, `QdrantVectorStore`, `PineconeVectorStore` now importable from `synapsekit` and `synapsekit.retrieval` via lazy imports
- **File existence checks** — `PDFLoader`, `HTMLLoader`, `CSVLoader`, `JSONLoader` now raise `FileNotFoundError` with a clear message before attempting to read
- **Parameter validation** — `FunctionCallingAgent` and `ReActAgent` reject `max_iterations < 1`; `ConversationMemory` rejects `window < 1`

### Fixed

- Loader import-error tests now use temp files to work correctly with file existence checks

### Stats

- 357 tests passing (was 332)

---

## [0.5.0] — 2026-03-12

### Added

- **Text Splitters** — `BaseSplitter` ABC, `CharacterTextSplitter`, `RecursiveCharacterTextSplitter`, `TokenAwareSplitter`, `SemanticSplitter` (cosine similarity boundaries)
- **Function calling** — `call_with_tools()` added to `GeminiLLM` and `MistralLLM` (now 4 providers support native tool use)
- **LLM Caching** — `AsyncLRUCache` with SHA-256 cache keys, opt-in via `LLMConfig(cache=True)`
- **LLM Retries** — exponential backoff via `retry_async()`, skips auth errors, opt-in via `LLMConfig(max_retries=N)`
- **Graph Cycles** — `compile(allow_cycles=True)` skips static cycle detection for intentional loops
- **Configurable max_steps** — `compile(max_steps=N)` overrides the default `_MAX_STEPS=100` guard
- **Graph Checkpointing** — `BaseCheckpointer` ABC, `InMemoryCheckpointer`, `SQLiteCheckpointer`
- `CompiledGraph.resume(graph_id, checkpointer)` — re-execute from saved state
- Adjacency index optimization for `CompiledGraph._next_wave()`
- `RAGConfig.splitter` — plug any `BaseSplitter` into the RAG pipeline
- `TextSplitter` alias preserved for backward compatibility
- 65 new tests (332 total)

### Changed

- `LLMConfig` gains `cache`, `cache_maxsize`, `max_retries`, `retry_delay` fields (all off by default)
- `pyproject.toml` description updated
- Version bumped to `0.5.0`

---

## [0.4.0] — 2026-03-11

### Added

- **Graph Workflows** — `StateGraph` fluent builder with compile-time validation and DFS cycle detection
- **`CompiledGraph`** — wave-based async executor with `run()`, `stream()`, and `run_sync()`
- **`Node`**, **`Edge`**, **`ConditionalEdge`** — sync and async node functions, static and conditional routing
- **`agent_node()`**, **`rag_node()`** — wrap `AgentExecutor` or `RAGPipeline` as graph nodes
- **Parallel execution** — nodes in the same wave run concurrently via `asyncio.gather()`
- **Mermaid export** — `CompiledGraph.get_mermaid()` returns a flowchart string
- **`_MAX_STEPS = 100`** guard against infinite conditional loops
- **`GraphConfigError`**, **`GraphRuntimeError`** — distinct error types for build vs runtime failures
- 44 new tests (267 total)

### Changed

- **Build tooling** migrated from Poetry to [uv](https://github.com/astral-sh/uv)
- `pyproject.toml` updated to PEP 621 `[project]` format with hatchling build backend
- Version bumped to `0.4.0`

---

## [0.3.0] — 2026-03-10

### Added

- **`BaseTool` ABC** — `run()`, `schema()`, `anthropic_schema()`, `ToolResult`
- **`ToolRegistry`** — tool lookup by name, OpenAI and Anthropic schema generation
- **`AgentMemory`** — step scratchpad with `format_scratchpad()` and `max_steps` limit
- **`ReActAgent`** — Thought → Action → Observation loop, works with any `BaseLLM`
- **`FunctionCallingAgent`** — native OpenAI `tool_calls` and Anthropic `tool_use`, multi-tool per step
- **`AgentExecutor`** — unified runner with `run()`, `stream()`, `run_sync()`, auto-selects agent type
- **`call_with_tools()`** — added to `OpenAILLM` and `AnthropicLLM`
- **Built-in tools**: `CalculatorTool`, `PythonREPLTool`, `FileReadTool`, `WebSearchTool`, `SQLQueryTool`
- 82 new tests (223 total)

---

## [0.2.0] — 2026-03-08

### Added

- **Loaders**: `PDFLoader`, `HTMLLoader`, `CSVLoader`, `JSONLoader`, `DirectoryLoader`, `WebLoader`
- **Output parsers**: `JSONParser`, `PydanticParser`, `ListParser`
- **Vector store backends**: `ChromaVectorStore`, `FAISSVectorStore`, `QdrantVectorStore`, `PineconeVectorStore`
- **LLM providers**: `OllamaLLM`, `CohereLLM`, `MistralLLM`, `GeminiLLM`, `BedrockLLM`
- **Prompt templates**: `PromptTemplate`, `ChatPromptTemplate`, `FewShotPromptTemplate`
- **`VectorStore` ABC** — unified interface for all backends
- `Retriever.add()` — cleaner public API
- `RAGPipeline.add_documents(docs)` — ingest `List[Document]` directly
- `RAG.add_documents()` and `RAG.add_documents_async()`
- 89 new tests (141 total)

---

## [0.1.0] — 2026-03-05

### Added

- **`BaseLLM` ABC** and `LLMConfig`
- **`OpenAILLM`** — async streaming
- **`AnthropicLLM`** — async streaming
- **`SynapsekitEmbeddings`** — sentence-transformers backend
- **`InMemoryVectorStore`** — numpy cosine similarity with `.npz` persistence
- **`Retriever`** — vector search with optional BM25 reranking
- **`TextSplitter`** — pure Python, zero dependencies
- **`ConversationMemory`** — sliding window
- **`TokenTracer`** — tokens, latency, and cost per call
- **`TextLoader`**, **`StringLoader`**
- **`RAGPipeline`** — full retrieval-augmented generation orchestrator
- **`RAG`** facade — 3-line happy path
- **`run_sync()`** — works inside and outside running event loops
- 52 tests
