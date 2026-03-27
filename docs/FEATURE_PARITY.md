# SynapseKit vs LangChain — Feature Parity Report

> Updated for v1.4.1 (2026-03-27)

## Phase 1: RAG Pipelines

| Capability | LangChain | SynapseKit | Gap |
|---|---|---|---|
| Document loaders | 200+ (PDF, Notion, Slack, Google Drive...) | 14 (Text, String, PDF, HTML, CSV, JSON, Directory, Web, Excel, PowerPoint, Docx, Markdown, Contextual, SentenceWindow) | Covers all common formats |
| Text splitters | 5+ strategies (recursive, semantic, token-aware) | 6 (character, recursive, token-aware, semantic, markdown, markdown-header) | At parity |
| Vector stores | 20+ (Chroma, FAISS, Pinecone, Weaviate, PGVector...) | 5 (InMemory, Chroma, FAISS, Qdrant, Pinecone) | Solid — covers the major ones |
| Retrieval strategies | Similarity, MMR, hybrid, self-query, ensemble, CRAG, compression | 20 strategies (vector+BM25, MMR, RAG Fusion, Contextual, SentenceWindow, SelfQuery, ParentDoc, CrossEncoder, CRAG, QueryDecomp, Compression, Ensemble, CohereRerank, StepBack, FLARE, HyDE, HybridSearch, SelfRAG, AdaptiveRAG, MultiStep, GraphRAG) | Exceeds LangChain |
| Conversation memory | 4+ types (buffer, summary, window, entity) | 9 (ConversationMemory, HybridMemory, SQLiteConversationMemory, RedisConversationMemory, SummaryBufferMemory, TokenBufferMemory, BufferMemory, EntityMemory) | Exceeds LangChain |
| Streaming | Yes | Yes (stream-first) | At parity |
| HyDE, multi-query | Yes | RAG Fusion (multi-query + RRF), QueryDecomposition, HyDE | At parity |

**Verdict:** Excellent. 14 loaders, 6 splitters, 20 retrieval strategies, 9 memory backends. Exceeds LangChain retrieval coverage with FLARE, Step-Back, Cohere reranking, Self-RAG, Adaptive RAG, Multi-Step, and GraphRAG retrieval.

---

## Phase 2: LLM Providers

| Capability | LangChain | SynapseKit | Gap |
|---|---|---|---|
| Providers | 38+ | 25 (OpenAI, Anthropic, Ollama, Cohere, Mistral, Gemini, Vertex AI, Bedrock, Azure OpenAI, Groq, DeepSeek, OpenRouter, Together, Fireworks, Perplexity, Cerebras, Moonshot, Zhipu, Cloudflare, AI21, Databricks, ERNIE, llama.cpp, Minimax, Aleph Alpha) | Covers all major ones |
| Unified interface | Yes (invoke/stream/batch) | Yes (generate/stream) | At parity |
| Auto-detect from model name | No (explicit class) | Yes | SynapseKit advantage |
| Caching | Built-in (memory, SQLite, Redis) | In-memory LRU + SQLite + Filesystem + Redis | At parity |
| Rate limiting | Built-in | Token-bucket (`requests_per_minute`) | At parity |
| Retries | Built-in | Exponential backoff (`max_retries`) | At parity |
| Structured output | Pydantic output parsers | `generate_structured()` with retry | At parity |
| Callbacks / observability | LangSmith integration | TokenTracer + `ExecutionTrace` + `OTelExporter` + `TracingMiddleware` + `TracingUI` + `DistributedTracer` | At parity |

**Verdict:** Excellent coverage. 23 providers cover 99%+ of real usage. Caching (memory + SQLite + filesystem + Redis), retries, rate limiting, structured output, and full observability stack (OTel, tracing UI, distributed tracing) all done. Local inference via llama.cpp requires no API key.

---

## Phase 3: Agents

| Capability | LangChain | SynapseKit | Gap |
|---|---|---|---|
| ReAct agent | Yes | Yes | At parity |
| Function calling agent | Yes (any provider with tool support) | Yes (OpenAI, Anthropic, Gemini, Mistral) | 4 providers — missing Cohere |
| Built-in tools | 50+ (search, code, DB, APIs, web) | 38 (Calculator, PythonREPL, FileRead, FileWrite, FileList, WebSearch, DuckDuckGoSearch, SQL, HTTP, GraphQL, DateTime, Regex, JSONQuery, HumanInput, Wikipedia, Summarization, SentimentAnalysis, Translation, WebScraper, Shell, SQLSchemaInspection, PDFReader, ArxivSearch, TavilySearch, Email, GitHubAPI, PubMedSearch, VectorSearch, YouTubeSearch, Slack, Jira, BraveSearch, APIBuilder, GoogleCalendar, AWSLambda, ImageAnalysis, TextToSpeech, SpeechToText) | Fewer but covers essentials |
| Custom tools | @tool decorator + StructuredTool | @tool decorator + BaseTool subclass | At parity |
| Streaming agent steps | Yes | Yes | At parity |
| Human input tool | Yes | Yes (`HumanInputTool`) | At parity |
| Multi-agent orchestration | Yes (via LangGraph) | Yes (Supervisor, Handoff, Crew) | At parity |
| Tool sandboxing/timeout | Partial | ShellTool has timeout + allowed-commands | Partial parity |

**Verdict:** Excellent. `@tool` decorator, function calling on 4 providers, 38 built-in tools (including ImageAnalysis, TextToSpeech, SpeechToText, APIBuilder, GoogleCalendar, AWSLambda), MCP client/server, 3 multi-agent patterns (Supervisor, Handoff, Crew), guardrails (content filter, PII detector, topic restrictor), and A2A protocol support. At parity with LangChain/LangGraph.

---

## Phase 4: Graph Workflows

| Capability | LangChain (LangGraph) | SynapseKit | Gap |
|---|---|---|---|
| StateGraph builder | Yes | Yes | At parity |
| Conditional routing | Yes | Yes | At parity |
| Parallel execution | Yes (asyncio.gather) | Yes (asyncio.gather) | At parity |
| Mermaid export | Yes | Yes + trace highlighting + GraphVisualizer | Exceeds LangGraph |
| Streaming | Node + token level | Node + token level (`llm_node` + `stream_tokens`) + SSE (`sse_stream`) + WebSocket (`ws_stream`) | At parity |
| Cyclic graphs (loops) | Yes | Yes (`compile(allow_cycles=True)`) | At parity |
| Human-in-the-loop | interrupt() + Command(resume=) | `GraphInterrupt` + `resume(updates=...)` + `approval_node()` | At parity |
| Dynamic routing | Conditional edges | `ConditionalEdge` + `dynamic_route_node()` | At parity |
| Checkpointing / persistence | SQLite, Postgres, Redis | InMemory + SQLite + JSON file + Redis + Postgres | At parity |
| Subgraphs | Yes | Yes (`subgraph_node()`, `fan_out_node()`) | At parity |
| Typed state + reducers | Annotated types with reducers | `TypedState` + `StateField` with per-field reducers | At parity |
| Event callbacks | Yes | `EventHooks` (node_start, node_complete, wave_start, wave_complete) | At parity |

**Verdict:** At parity across all features. Human-in-the-loop (with `approval_node`), dynamic routing, subgraphs, fan-out/fan-in, typed state with reducers, token streaming, SSE + WebSocket streaming, event callbacks, execution tracing, cycles, and checkpointing (InMemory, SQLite, JSON file, Redis, Postgres) all done.

---

## Phase 5: Evaluation & Observability

| Capability | LangChain / LangSmith | SynapseKit | Gap |
|---|---|---|---|
| RAG evaluation metrics | RAGAS integration | Built-in faithfulness, relevancy, groundedness | At parity |
| Evaluation pipeline | LangSmith (hosted, paid) | `EvaluationPipeline` + `EvaluationResult` | At parity |
| Eval test cases | LangSmith datasets | `@eval_case` decorator — mark functions with thresholds | SynapseKit advantage (local, no SaaS) |
| CLI eval runner | LangSmith UI | `synapsekit test` — threshold enforcement, JSON/table output | SynapseKit advantage (local, no SaaS) |
| OpenTelemetry tracing | LangSmith (proprietary) | `OTelExporter` + `Span` + `TracingMiddleware` | SynapseKit advantage (open standard) |
| Tracing UI | LangSmith dashboard (hosted) | `TracingUI` — HTML dashboard, file, local HTTP server | At parity (self-hosted) |
| Distributed tracing | LangSmith | `DistributedTracer` + `TraceSpan` — parent-child spans | At parity |

**Verdict:** At parity with LangSmith for evaluation and observability. SynapseKit's `@eval_case` + `synapsekit test` CLI runs evaluations locally without a SaaS dependency, and OTel tracing uses an open standard rather than a proprietary platform.

---

## Phase 6: Deployment & Ops

| Capability | LangChain / LangServe | SynapseKit | Gap |
|---|---|---|---|
| One-command deployment | LangServe (`langchain serve`) | `synapsekit serve` — auto-detects RAG/Graph/Agent, wraps as FastAPI | At parity |
| Cost tracking | External / manual | `CostTracker` — hierarchical attribution with scope context manager | SynapseKit advantage |
| Budget enforcement | None built-in | `BudgetGuard` — per-request, per-user, daily limits with circuit breaker | SynapseKit advantage |
| Prompt management | LangChain Hub (hosted) | `PromptHub` — local filesystem registry with versioning | At parity (self-hosted) |
| Plugin / extension system | LangChain integrations (monorepo) | `PluginRegistry` — entry-point plugin system for community extensions | At parity |

**Verdict:** SynapseKit v1.2.0 matches LangServe for deployment and exceeds LangChain for cost intelligence. `CostTracker` and `BudgetGuard` provide hierarchical cost attribution and budget circuit-breakers that LangChain does not offer natively — capabilities that LangChain users typically pay for via LangSmith or third-party tooling.

---

## Overall Assessment

| | LangChain | SynapseKit | Notes |
|---|---|---|---|
| Breadth | Massive (200+ loaders, 38+ providers, 50+ tools) | Focused (14 loaders, 23 providers, 38 tools) | SynapseKit covers the 80/20 |
| API simplicity | Complex, lots of boilerplate | Clean, 3-line happy path | SynapseKit advantage |
| Async/streaming | Retrofitted | Native from day 1 | SynapseKit advantage |
| Dependencies | Heavy (langchain-core + per-provider) | 2 hard deps | SynapseKit advantage |
| Production features | Caching, retries, rate limiting, observability | Caching, retries, rate limiting, structured output, OTel tracing, tracing UI, guardrails | At parity |
| Graph workflows | Mature (HITL, checkpoints, cycles, subgraphs, typed state) | HITL, checkpoints (InMemory, SQLite, JSON, Redis, Postgres), cycles, subgraphs, typed state, fan-out, SSE, WebSocket, event callbacks, execution tracing | At parity |
| Multi-agent | LangGraph supervisor, handoff | Supervisor, Handoff, Crew + MCP client/server | At parity |
| Evaluation | LangSmith (hosted, paid) | Built-in metrics + `@eval_case` + `synapsekit test` CLI (local, free) | SynapseKit advantage |
| Cost intelligence | External / LangSmith | `CostTracker` + `BudgetGuard` (hierarchical attribution, circuit breaker) | SynapseKit advantage |
| Deployment | LangServe | `synapsekit serve` (auto-detect RAG/Graph/Agent, FastAPI) | At parity |
| Prompt management | LangChain Hub (hosted) | `PromptHub` (local, versioned) | At parity |
| Plugin system | LangChain integrations | `PluginRegistry` (entry-point based) | At parity |
| Retrieval | 10+ strategies | 20 strategies | Exceeds LangChain |
| Memory | 4+ types | 9 types (window, hybrid, SQLite, Redis, summary buffer, token buffer, buffer, entity) | Exceeds LangChain |
| Multimodal | Image, audio inputs | ImageContent, AudioContent, MultimodalMessage, ImageLoader | At parity |
| Interoperability | Hub integrations | MCP protocol + A2A protocol | Modern standards |

### Where SynapseKit wins

- Simpler API, less boilerplate
- Truly async-native and streaming-first
- Minimal dependencies (2 hard deps)
- Auto-detection of providers from model name
- 23 LLM providers (including local llama.cpp, no API key required), 14 loaders, 38 tools — covers real-world needs
- 20 retrieval strategies exceeding LangChain
- Graph workflows at full parity with LangGraph including Postgres + Redis checkpoints
- MCP + A2A protocol for modern interoperability
- Multi-agent orchestration (Supervisor, Handoff, Crew)
- Built-in evaluation metrics (faithfulness, relevancy, groundedness)
- `@eval_case` + `synapsekit test` CLI — local eval runner without LangSmith SaaS dependency
- `CostTracker` + `BudgetGuard` — cost intelligence and budget enforcement not available in LangChain
- `synapsekit serve` — one-command FastAPI deployment with auto-detection
- `PromptHub` — local versioned prompt registry without LangChain Hub dependency
- `PluginRegistry` — entry-point plugin system for community extensions
- Full observability stack (OTel, tracing UI, distributed tracing)
- Agent guardrails (content filter, PII, topic restriction)
- Multimodal support (image, audio) with provider format conversion
- API stability markers (@public_api, @experimental, @deprecated)

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
35. CohereReranker — Cohere Rerank API reranking (v0.6.5)
36. StepBackRetriever — step-back question generation + parallel retrieval (v0.6.5)
37. FLARERetriever — Forward-Looking Active REtrieval with iterative search markers (v0.6.5)
38. DuckDuckGoSearchTool, PDFReaderTool, GraphQLTool — 3 new tools (v0.6.5)
39. TokenBufferMemory — token-budget-aware memory without LLM (v0.6.5)
40. RedisLLMCache — distributed Redis cache backend (v0.6.5)
41. PerplexityLLM — Perplexity AI with Sonar models (v0.6.6)
42. CerebrasLLM — Cerebras ultra-fast inference (v0.6.6)
43. HybridSearchRetriever — BM25 + vector RRF fusion (v0.6.6)
44. SelfRAGRetriever — self-reflective retrieve-grade-generate-check loop (v0.6.6)
45. AdaptiveRAGRetriever — LLM query classification routing (v0.6.6)
46. MultiStepRetriever — iterative gap-fill retrieval (v0.6.6)
47. ArxivSearchTool, TavilySearchTool — 2 new search tools (v0.6.6)
48. BufferMemory — simple unbounded buffer (v0.6.6)
49. EntityMemory — LLM-based entity extraction and tracking (v0.6.6)
50. Python `>=3.10` requirement (v0.6.7)
51. EmailTool — SMTP email sending with env-var config (v0.6.8)
52. GitHubAPITool — GitHub REST API search/get (stdlib only) (v0.6.8)
53. PubMedSearchTool — biomedical literature search via NCBI (v0.6.8)
54. VectorSearchTool — wrap any Retriever as an agent tool (v0.6.8)
55. YouTubeSearchTool — YouTube video search (v0.6.8)
56. ExecutionTrace + TraceEntry — graph execution tracing with timing (v0.6.8)
57. `ws_stream()` — WebSocket streaming for graph execution (v0.6.8)
58. SlackTool — Slack messaging via webhook or bot token (v0.6.9)
59. JiraTool — Jira REST API v2 (search, get, create, comment) (v0.6.9)
60. BraveSearchTool — Brave Search API web search (v0.6.9)
61. `approval_node()` — gate graph on human approval via GraphInterrupt (v0.6.9)
62. `dynamic_route_node()` — runtime subgraph routing (v0.6.9)
63. `MCPClient` + `MCPToolAdapter` — MCP client with stdio/SSE transport (v0.7.0)
64. `MCPServer` — expose SynapseKit tools via MCP protocol (v0.7.0)
65. `SupervisorAgent` + `WorkerAgent` — supervisor multi-agent pattern (v0.7.0)
66. `HandoffChain` + `Handoff` — condition-based agent handoff (v0.7.0)
67. `Crew` + `CrewAgent` + `Task` — role-based multi-agent teams (v0.7.0)
68. `FaithfulnessMetric` — claim verification against sources (v0.8.0)
69. `RelevancyMetric` — document relevance scoring (v0.8.0)
70. `GroundednessMetric` — answer grounding score (v0.8.0)
71. `EvaluationPipeline` — multi-metric evaluation runner (v0.8.0)
72. `OTelExporter` + `Span` + `TracingMiddleware` — OpenTelemetry tracing (v0.8.0)
73. `TracingUI` — HTML trace visualization dashboard (v0.8.0)
74. `A2AClient` + `A2AServer` + `AgentCard` — Google A2A protocol (v0.9.0)
75. `ContentFilter` + `PIIDetector` + `TopicRestrictor` + `Guardrails` — agent guardrails (v0.9.0)
76. `DistributedTracer` + `TraceSpan` — distributed tracing with parent-child spans (v0.9.0)
77. `ImageContent` + `AudioContent` + `MultimodalMessage` — multimodal support (v1.0.0)
78. `ImageLoader` — image loading with optional vision LLM description (v1.0.0)
79. `@public_api`, `@experimental`, `@deprecated` — API stability markers (v1.0.0)
80. `GraphRAGRetriever` + `KnowledgeGraph` — knowledge-graph-augmented retrieval (v1.1.0)
81. `RedisConversationMemory` — Redis-backed conversation memory with windowing (v1.1.0)
82. `VertexAILLM` — Google Vertex AI provider with ADC and function calling (v1.1.0)
83. `MarkdownTextSplitter` — header-aware markdown splitting with context preservation (v1.1.0)
84. `GraphVisualizer` + `get_mermaid_with_trace()` — graph visualization and execution replay (v1.1.0)
85. `synapsekit serve` — one-command FastAPI deployment with RAG/Graph/Agent auto-detection (v1.2.0)
86. `CostTracker` — hierarchical cost attribution with scope context manager (v1.2.0)
87. `BudgetGuard` — per-request, per-user, daily budget limits with circuit breaker (v1.2.0)
88. `@eval_case` decorator — mark functions as eval test cases with thresholds (v1.2.0)
89. `synapsekit test` — CLI eval runner with threshold enforcement and JSON/table output (v1.2.0)
90. `PromptHub` — local filesystem prompt registry with versioning (v1.2.0)
91. `PluginRegistry` — entry-point plugin system for community extensions (v1.2.0)
92. `RedisCheckpointer` + `PostgresCheckpointer` — graph checkpoint backends (v1.2.0)
93. `CostRouter` — route to cheapest model meeting quality threshold (v1.3.0)
94. `FallbackChain` — cascade through models on failure (v1.3.0)
95. `PIIRedactor` — redact PII from text before LLM calls (v1.3.0)
96. `AuditLog` — tamper-evident JSONL audit trail for agent actions (v1.3.0)
97. `AudioLoader` + `VideoLoader` — transcribe audio/video files as documents (v1.3.0)
98. `EvalRegressionTracker` — track eval metric regressions across versions (v1.3.0)
99. `MoonshotLLM` — Moonshot AI (Kimi) provider (v1.3.0)
100. `ZhipuLLM` — Zhipu AI GLM provider (v1.3.0)
101. `CloudflareLLM` — Cloudflare Workers AI provider (v1.3.0)
102. `APIBuilderTool` — build and execute API calls from OpenAPI specs or intent (v1.4.0)
103. `GoogleCalendarTool` — create, list, delete Google Calendar events (v1.4.0)
104. `AWSLambdaTool` — invoke AWS Lambda functions with standard boto3 credentials (v1.4.0)
105. `AI21LLM` — AI21 Jamba models with 256K context (v1.4.0)
106. `DatabricksLLM` — Databricks Foundation Model APIs via OpenAI-compatible endpoint (v1.4.0)
107. `ErnieLLM` — Baidu ERNIE Bot Chinese-English bilingual models (v1.4.0)
108. `LlamaCppLLM` — local GGUF models via llama-cpp-python, no API key (v1.4.0)
109. `ImageAnalysisTool` — analyze images with any multimodal LLM (v1.4.0)
110. `TextToSpeechTool` — text → speech audio via OpenAI TTS (v1.4.0)
111. `SpeechToTextTool` — transcribe audio with Whisper API or local model (v1.4.0)

### Remaining priority gaps

1. Streaming agent steps (token-level per reasoning step)
2. Advanced graph features (time-travel debugging, branch/merge)
3. EU compliance extensions (`DataResidency`, `ConsentManager`, GDPR toolkit)
