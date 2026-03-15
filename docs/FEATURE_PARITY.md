# SynapseKit vs LangChain — Feature Parity Report

> Updated for v0.6.4 (2026-03-15)

## Phase 1: RAG Pipelines

| Capability | LangChain | SynapseKit | Gap |
|---|---|---|---|
| Document loaders | 200+ (PDF, Notion, Slack, Google Drive...) | 14 (Text, String, PDF, HTML, CSV, JSON, Directory, Web, Excel, PowerPoint, Docx, Markdown, Contextual, SentenceWindow) | Covers all common formats |
| Text splitters | 5+ strategies (recursive, semantic, token-aware) | 5 (character, recursive, token-aware, semantic, markdown) | At parity |
| Vector stores | 20+ (Chroma, FAISS, Pinecone, Weaviate, PGVector...) | 5 (InMemory, Chroma, FAISS, Qdrant, Pinecone) | Solid — covers the major ones |
| Retrieval strategies | Similarity, MMR, hybrid, self-query, ensemble, CRAG, compression | 10 strategies (vector+BM25, MMR, RAG Fusion, Contextual, SentenceWindow, SelfQuery, ParentDoc, CrossEncoder, CRAG, QueryDecomp, Compression, Ensemble) | At parity |
| Conversation memory | 4+ types (buffer, summary, window, entity) | 4 (ConversationMemory, HybridMemory, SQLiteConversationMemory, SummaryBufferMemory) | At parity |
| Streaming | Yes | Yes (stream-first) | At parity |
| HyDE, multi-query | Yes | RAG Fusion (multi-query + RRF), QueryDecomposition, HyDE | At parity |

**Verdict:** Excellent. 14 loaders, 5 splitters, 11 retrieval strategies, 5 memory backends. Covers 95%+ of real use cases. At parity with LangChain retrieval.

---

## Phase 2: LLM Providers

| Capability | LangChain | SynapseKit | Gap |
|---|---|---|---|
| Providers | 38+ | 13 (OpenAI, Anthropic, Ollama, Cohere, Mistral, Gemini, Bedrock, Azure OpenAI, Groq, DeepSeek, OpenRouter, Together, Fireworks) | Covers all major ones |
| Unified interface | Yes (invoke/stream/batch) | Yes (generate/stream) | At parity |
| Auto-detect from model name | No (explicit class) | Yes | SynapseKit advantage |
| Caching | Built-in (memory, SQLite, Redis) | In-memory LRU + SQLite + Filesystem | Missing Redis backend |
| Rate limiting | Built-in | Token-bucket (`requests_per_minute`) | At parity |
| Retries | Built-in | Exponential backoff (`max_retries`) | At parity |
| Structured output | Pydantic output parsers | `generate_structured()` with retry | At parity |
| Callbacks / observability | LangSmith integration | TokenTracer only | Basic vs enterprise |

**Verdict:** Excellent coverage. 13 providers cover 99%+ of real usage. Caching (memory + SQLite + filesystem), retries, rate limiting, and structured output all done. Missing Redis cache and deep observability.

---

## Phase 3: Agents

| Capability | LangChain | SynapseKit | Gap |
|---|---|---|---|
| ReAct agent | Yes | Yes | At parity |
| Function calling agent | Yes (any provider with tool support) | Yes (OpenAI, Anthropic, Gemini, Mistral) | 4 providers — missing Cohere |
| Built-in tools | 50+ (search, code, DB, APIs, web) | 19 (Calculator, PythonREPL, FileRead, FileWrite, FileList, WebSearch, SQL, HTTP, DateTime, Regex, JSONQuery, HumanInput, Wikipedia, Summarization, SentimentAnalysis, Translation, WebScraper, Shell, SQLSchemaInspection) | Fewer but covers essentials |
| Custom tools | @tool decorator + StructuredTool | @tool decorator + BaseTool subclass | At parity |
| Streaming agent steps | Yes | Yes | At parity |
| Human input tool | Yes | Yes (`HumanInputTool`) | At parity |
| Multi-agent orchestration | Yes (via LangGraph) | No | Missing |
| Tool sandboxing/timeout | Partial | ShellTool has timeout + allowed-commands | Partial parity |

**Verdict:** Strong for single-agent workflows. `@tool` decorator, function calling on 4 providers, 19 built-in tools including shell, SQL schema inspection, and web scraper. Missing multi-agent orchestration.

---

## Phase 4: Graph Workflows

| Capability | LangChain (LangGraph) | SynapseKit | Gap |
|---|---|---|---|
| StateGraph builder | Yes | Yes | At parity |
| Conditional routing | Yes | Yes | At parity |
| Parallel execution | Yes (asyncio.gather) | Yes (asyncio.gather) | At parity |
| Mermaid export | Yes | Yes | At parity |
| Streaming | Node + token level | Node + token level (`llm_node` + `stream_tokens`) + SSE (`sse_stream`) | At parity |
| Cyclic graphs (loops) | Yes | Yes (`compile(allow_cycles=True)`) | At parity |
| Human-in-the-loop | interrupt() + Command(resume=) | `GraphInterrupt` + `resume(updates=...)` | At parity |
| Checkpointing / persistence | SQLite, Postgres, Redis | InMemory + SQLite + JSON file | Missing Postgres/Redis backends |
| Subgraphs | Yes | Yes (`subgraph_node()`, `fan_out_node()`) | At parity |
| Typed state + reducers | Annotated types with reducers | `TypedState` + `StateField` with per-field reducers | At parity |
| Event callbacks | Yes | `EventHooks` (node_start, node_complete, wave_start, wave_complete) | At parity |

**Verdict:** At parity for core features. Human-in-the-loop, subgraphs, fan-out/fan-in, typed state with reducers, token streaming, SSE streaming, event callbacks, cycles, and checkpointing (InMemory, SQLite, JSON file) all done. Only remaining gap: Postgres/Redis checkpoint backends.

---

## Overall Assessment

| | LangChain | SynapseKit | Notes |
|---|---|---|---|
| Breadth | Massive (200+ loaders, 38+ providers, 50+ tools) | Focused (14 loaders, 13 providers, 19 tools) | SynapseKit covers the 80/20 |
| API simplicity | Complex, lots of boilerplate | Clean, 3-line happy path | SynapseKit advantage |
| Async/streaming | Retrofitted | Native from day 1 | SynapseKit advantage |
| Dependencies | Heavy (langchain-core + per-provider) | 2 hard deps | SynapseKit advantage |
| Production features | Caching, retries, rate limiting, observability | Caching (memory+SQLite+filesystem), retries, rate limiting, structured output | Close — missing Redis cache, deep observability |
| Graph workflows | Mature (HITL, checkpoints, cycles, subgraphs, typed state) | HITL, checkpoints, cycles, subgraphs, typed state, fan-out, SSE, event callbacks | At parity |
| Retrieval | 10+ strategies | 11 strategies | At parity |
| Memory | 4+ types | 5 types (window, hybrid, SQLite, summary buffer) | At parity |

### Where SynapseKit already wins

- Simpler API, less boilerplate
- Truly async-native and streaming-first
- Minimal dependencies (2 hard deps)
- Auto-detection of providers from model name
- 13 LLM providers, 14 loaders, 19 tools — covers real-world needs
- 11 retrieval strategies including CRAG, ensemble, compression, and HyDE
- Graph workflows at feature parity with LangGraph

### Closed since v0.5.0

1. `@tool` decorator for quick tool creation (v0.5.1)
2. Metadata filtering in vector stores (v0.5.1)
3. MMR retrieval for diversity-aware search (v0.5.2)
4. Rate limiting with token-bucket algorithm (v0.5.2)
5. Structured output with retry (v0.5.2)
6. Cache hit/miss statistics (v0.5.2)
7. RAG Fusion — multi-query + Reciprocal Rank Fusion (v0.5.3)
8. 3 new LLM providers: Azure OpenAI, Groq, DeepSeek (v0.5.3)
9. SQLite persistent LLM cache (v0.5.3)
10. Excel and PowerPoint loaders (v0.5.3)
11. 6 new built-in tools: HTTP, FileWrite, FileList, DateTime, Regex, JSONQuery (v0.6.0)
12. 3 new LLM providers: OpenRouter, Together, Fireworks (v0.6.0)
13. Contextual Retrieval + Sentence Window retrieval (v0.6.0)
14. Human-in-the-loop with GraphInterrupt + resume (v0.6.1)
15. Subgraphs via `subgraph_node()` (v0.6.1)
16. Token-level streaming via `llm_node()` + `stream_tokens()` (v0.6.1)
17. SelfQuery, ParentDocument, CrossEncoder retrieval (v0.6.1)
18. HybridMemory — window + LLM summary (v0.6.1)
19. CRAG, Query Decomposition, Contextual Compression, Ensemble retrieval (v0.6.2)
20. SQLite persistent conversation memory (v0.6.2)
21. Summary Buffer memory (v0.6.2)
22. HumanInput + Wikipedia tools (v0.6.2)
23. Typed state with per-field reducers (v0.6.3)
24. Fan-out/fan-in parallel subgraph execution (v0.6.3)
25. SSE streaming for graph execution (v0.6.3)
26. Event callbacks/hooks for graph monitoring (v0.6.3)
27. Semantic LLM cache with embedding similarity (v0.6.3)
28. Summarization, SentimentAnalysis, Translation tools (v0.6.3)
29. DocxLoader, MarkdownLoader (v0.6.4)
30. HyDE retrieval (v0.6.4)
31. ShellTool, SQLSchemaInspectionTool, WebScraperTool (v0.6.4)
32. Filesystem LLM cache backend (v0.6.4)
33. JSON file checkpointer (v0.6.4)
34. TokenTracer COST_TABLE update — GPT-4.1, o3, Gemini 2.5, DeepSeek, Groq models (v0.6.4)

### Remaining priority gaps

1. Multi-agent orchestration
2. Multi-modal support (image inputs)
3. Evaluation framework (RAGAS-style metrics)
4. Deep observability (LangSmith equivalent)
