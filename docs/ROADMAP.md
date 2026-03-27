# SynapseKit Roadmap

## v0.5.0 — Production Features

- [x] Text Splitters (character, recursive, token-aware, semantic)
- [x] Function calling for Gemini and Mistral
- [x] LLM response caching (LRU, SHA-256 keys)
- [x] LLM retries with exponential backoff
- [x] Graph cycle support (`allow_cycles=True`)
- [x] Configurable `max_steps` for graph execution
- [x] Graph checkpointing (InMemory, SQLite)
- [x] `RAGConfig.splitter` — pluggable text splitters in RAG pipeline

## v0.5.1 — Polish

- [x] `@tool` decorator — create agent tools from plain functions with auto-generated JSON Schema
- [x] Metadata filtering — `VectorStore.search(metadata_filter={"key": "value"})`
- [x] Vector store lazy exports — all backends importable from `synapsekit`
- [x] File existence checks — loaders raise `FileNotFoundError` before attempting to read
- [x] Parameter validation — agents and memory reject invalid config

## v0.5.2 — Quality of Life

- [x] `__repr__` methods on `StateGraph`, `CompiledGraph`, `RAGPipeline`, `ReActAgent`, `FunctionCallingAgent`
- [x] Empty document handling — `RAGPipeline.add()` silently skips empty text
- [x] Retry for `call_with_tools()` — `LLMConfig(max_retries=N)` applies to function calling
- [x] Cache hit/miss statistics — `BaseLLM.cache_stats` property
- [x] MMR retrieval — `search_mmr()` and `retrieve_mmr()` for diversity-aware retrieval
- [x] Rate limiting — `LLMConfig(requests_per_minute=N)` with token-bucket algorithm
- [x] Structured output with retry — `generate_structured(llm, prompt, schema=Model)` parses to Pydantic

## v0.5.3 — Provider Expansion

- [x] Azure OpenAI — `AzureOpenAILLM` for enterprise Azure deployments
- [x] Groq — `GroqLLM` for ultra-fast inference (Llama, Mixtral, Gemma)
- [x] DeepSeek — `DeepSeekLLM` with function calling support
- [x] SQLite LLM cache — persistent cache via `LLMConfig(cache_backend="sqlite")`
- [x] RAG Fusion — `RAGFusionRetriever` with multi-query + Reciprocal Rank Fusion
- [x] Excel loader — `ExcelLoader` for `.xlsx` files
- [x] PowerPoint loader — `PowerPointLoader` for `.pptx` files
- [x] 10 LLM providers, 10 document loaders, 415 tests passing

## v0.6.0 — Tools, Providers & Advanced Retrieval

- [x] 6 new built-in tools: `HTTPRequestTool`, `FileWriteTool`, `FileListTool`, `DateTimeTool`, `RegexTool`, `JSONQueryTool`
- [x] 3 new LLM providers: `OpenRouterLLM`, `TogetherLLM`, `FireworksLLM`
- [x] `ContextualRetriever` — Anthropic-style contextual retrieval
- [x] `SentenceWindowRetriever` — sentence-level embedding with window expansion
- [x] 13 LLM providers, 11 built-in tools, 12 document loaders, 452 tests passing

## v0.6.1 — Graph Power-ups & Advanced Retrieval

- [x] `GraphInterrupt` — human-in-the-loop pause/resume for graph workflows
- [x] `subgraph_node()` — nest compiled graphs as nodes in parent graphs
- [x] `llm_node()` + `stream_tokens()` — token-level streaming from graph nodes
- [x] `SelfQueryRetriever` — LLM-generated metadata filters
- [x] `ParentDocumentRetriever` — small-chunk search, full-doc return
- [x] `CrossEncoderReranker` — cross-encoder reranking for precision
- [x] `HybridMemory` — sliding window + LLM summary
- [x] 13 providers, 11 tools, 12 loaders, 6 retrieval strategies, 482 tests passing

## v0.6.2 — Retrieval Strategies, Memory & Tools (current)

- [x] `CRAGRetriever` — Corrective RAG: grade docs, rewrite query, retry
- [x] `QueryDecompositionRetriever` — break complex queries into sub-queries
- [x] `ContextualCompressionRetriever` — compress docs to relevant excerpts
- [x] `EnsembleRetriever` — fuse results from multiple retrievers via weighted RRF
- [x] `SQLiteConversationMemory` — persistent chat history in SQLite
- [x] `SummaryBufferMemory` — token-budget-aware progressive summarization
- [x] `HumanInputTool` — pause agent for user input
- [x] `WikipediaTool` — Wikipedia article search and summaries
- [x] 13 providers, 13 tools, 12 loaders, 10 retrieval strategies, 4 memory backends, 512 tests passing

## v0.6.3 — Typed State, Fan-Out, SSE & LLM Tools

- [x] `TypedState` + `StateField` — typed state with per-field reducers for parallel merge
- [x] `fan_out_node()` — parallel subgraph execution with custom merge
- [x] `sse_stream()` — SSE streaming for graph execution
- [x] `EventHooks` + `GraphEvent` — event callbacks for graph monitoring
- [x] `SemanticCache` — similarity-based LLM cache using embeddings
- [x] `SummarizationTool` — summarize text with LLM
- [x] `SentimentAnalysisTool` — sentiment analysis with LLM
- [x] `TranslationTool` — translate text with LLM
- [x] 13 providers, 16 tools, 12 loaders, 10 retrieval strategies, 4 memory backends, 540 tests passing

## v0.6.4 — Loaders, HyDE, Tools & Persistence

- [x] `DocxLoader` — Word document loading via `python-docx`
- [x] `MarkdownLoader` — Markdown loading with optional YAML frontmatter stripping
- [x] `HyDERetriever` — Hypothetical Document Embeddings retrieval strategy
- [x] `ShellTool` — shell command execution with timeout and allowed-commands filter
- [x] `SQLSchemaInspectionTool` — database schema inspection (list tables, describe columns)
- [x] `FilesystemLLMCache` — persistent JSON file-based LLM cache backend
- [x] `JSONFileCheckpointer` — JSON file-based graph checkpoint persistence
- [x] `TokenTracer` COST_TABLE — GPT-4.1, o3, o4-mini, Gemini 2.5, DeepSeek-V3/R1, Groq models
- [x] 13 providers, 19 tools, 14 loaders, 11 retrieval strategies, 5 memory backends, 587 tests passing

## v0.6.5 — Retrieval, Tools, Memory & Redis Cache

- [x] `CohereReranker` — rerank results using Cohere Rerank API
- [x] `StepBackRetriever` — step-back question generation + parallel retrieval
- [x] `FLARERetriever` — Forward-Looking Active REtrieval with iterative `[SEARCH: ...]` markers
- [x] `DuckDuckGoSearchTool` — extended DuckDuckGo search with text and news types
- [x] `PDFReaderTool` — read and extract text from PDF files with optional page selection
- [x] `GraphQLTool` — execute GraphQL queries against any endpoint
- [x] `TokenBufferMemory` — token-budget-aware memory that drops oldest messages (no LLM)
- [x] `RedisLLMCache` — distributed Redis cache backend (`pip install synapsekit[redis]`)
- [x] 13 providers, 22 tools, 14 loaders, 14 retrieval strategies, 4 cache backends, 6 memory backends, 642 tests passing

## v0.6.6 — Providers, Retrieval, Tools & Memory

- [x] `PerplexityLLM` — Perplexity AI with Sonar models, OpenAI-compatible
- [x] `CerebrasLLM` — Cerebras ultra-fast inference, OpenAI-compatible
- [x] `HybridSearchRetriever` — BM25 + vector similarity via Reciprocal Rank Fusion
- [x] `SelfRAGRetriever` — self-reflective RAG: retrieve, grade, generate, check support, retry
- [x] `AdaptiveRAGRetriever` — LLM classifies query complexity and routes to different retrievers
- [x] `MultiStepRetriever` — iterative retrieval-generation with gap identification
- [x] `ArxivSearchTool` — search arXiv for academic papers (stdlib only, no deps)
- [x] `TavilySearchTool` — AI-optimized web search via Tavily API
- [x] `BufferMemory` — simplest unbounded buffer, keeps all messages until cleared
- [x] `EntityMemory` — LLM-based entity extraction with running descriptions and eviction
- [x] 15 providers, 24 tools, 14 loaders, 18 retrieval strategies, 4 cache backends, 8 memory backends, 698 tests passing

## v0.6.7 — Python Version Bump

- [x] Require Python `>=3.10` (was `>=3.9`)
- [x] Added Python 3.14 classifier

## v0.6.8 — Tools, Tracing & WebSocket Streaming

- [x] `EmailTool` — send emails via SMTP with env-var config
- [x] `GitHubAPITool` — GitHub REST API search/get for repos and issues (stdlib only)
- [x] `PubMedSearchTool` — biomedical literature search via NCBI E-utilities (stdlib only)
- [x] `VectorSearchTool` — wrap any `Retriever` as an agent tool
- [x] `YouTubeSearchTool` — YouTube video search (`pip install synapsekit[youtube]`)
- [x] `ExecutionTrace` + `TraceEntry` — graph execution tracing with timing and summaries
- [x] `ws_stream()` — WebSocket streaming for graph execution (Starlette/FastAPI compatible)
- [x] 15 providers, 29 tools, 14 loaders, 18 retrieval strategies, 4 cache backends, 8 memory backends, 743 tests passing

## v0.6.9 — Tools & Graph Routing

- [x] `SlackTool` — send messages via Slack webhook or bot token (stdlib only)
- [x] `JiraTool` — Jira REST API v2: search, get, create issues, add comments (stdlib only)
- [x] `BraveSearchTool` — web search via Brave Search API (stdlib only)
- [x] `approval_node()` — gate graph execution on human approval via `GraphInterrupt`
- [x] `dynamic_route_node()` — route to subgraphs at runtime based on routing function
- [x] 15 providers, 32 tools, 14 loaders, 18 retrieval strategies, 4 cache backends, 8 memory backends, 795 tests passing

## v0.7.0 — MCP + Multi-Agent Orchestration

- [x] `MCPClient` — connect to MCP servers via stdio or SSE transport
- [x] `MCPToolAdapter` — wrap MCP tools as SynapseKit `BaseTool` instances
- [x] `MCPServer` — expose SynapseKit tools as MCP-compatible tools
- [x] `SupervisorAgent` + `WorkerAgent` — delegate tasks via DELEGATE/FINAL protocol
- [x] `HandoffChain` + `Handoff` — condition-based agent transfers
- [x] `Crew` + `CrewAgent` + `Task` — role-based multi-agent teams (sequential & parallel)
- [x] 15 providers, 32 tools, 14 loaders, 18 retrieval strategies, MCP client/server, 3 multi-agent patterns, 844 tests passing

## v0.8.0 — Evaluation Metrics + Observability

- [x] `FaithfulnessMetric` — verify answer claims against source contexts via LLM judge
- [x] `RelevancyMetric` — check document relevance to query
- [x] `GroundednessMetric` — score answer grounding in source documents (0-1)
- [x] `EvaluationPipeline` + `EvaluationResult` — run multiple metrics with mean_score
- [x] `OTelExporter` — lightweight tracing with optional OTLP export
- [x] `Span` — individual trace spans
- [x] `TracingMiddleware` — auto-trace LLM calls
- [x] `TracingUI` — render traces as HTML dashboard, file, or local HTTP server
- [x] 944 tests passing

## v0.9.0 — A2A Protocol, Guardrails, Distributed Tracing

- [x] `A2AClient` + `A2AServer` — Google Agent-to-Agent protocol support
- [x] `AgentCard` — agent capability discovery for A2A
- [x] `A2ATask`, `A2AMessage`, `TaskState` — A2A protocol types
- [x] `ContentFilter` — blocked patterns, words, max length
- [x] `PIIDetector` — email, phone, SSN, credit card, IP detection
- [x] `TopicRestrictor` — blocked topic enforcement
- [x] `Guardrails` — composite checker combining multiple guardrail types
- [x] `DistributedTracer` + `TraceSpan` — parent-child span relationships with events and timing
- [x] 1008 tests passing

## v1.0.0 — Multimodal, Image Loader, API Stability

- [x] `ImageContent` — from_file, from_url, from_base64 with OpenAI/Anthropic format conversion
- [x] `AudioContent` — from_file, from_base64
- [x] `MultimodalMessage` — combines text + images, converts to provider-specific formats
- [x] `ImageLoader` — sync/async image loading with optional vision LLM description
- [x] `@public_api` — mark stable public API surface
- [x] `@experimental` — FutureWarning on first use
- [x] `@deprecated(reason, alternative)` — DeprecationWarning with migration guidance
- [x] 1011 tests passing

## v1.1.0 — Retrieval, Memory, Providers & Visualization

- [x] `GraphRAGRetriever` — knowledge-graph-augmented retrieval: extract entities via LLM, traverse KG, merge with vector results
- [x] `KnowledgeGraph` — in-memory triple store with BFS traversal, entity-document linking, LLM-powered extraction
- [x] `RedisConversationMemory` — Redis-backed conversation memory with windowing, multi-conversation, JSON serialization
- [x] `VertexAILLM` — Google Vertex AI provider with ADC auth, streaming, native function calling
- [x] `MarkdownTextSplitter` — header-hierarchy-aware splitting with parent context preservation
- [x] `GraphVisualizer` — ASCII timeline, Mermaid trace highlighting, step replay, HTML export
- [x] `get_mermaid_with_trace()` — Mermaid diagrams with CSS status classes (completed/errored/skipped)
- [x] 16 providers, 20 retrieval strategies, 6 text splitters, 9 memory backends, 1047 tests passing

## v1.2.0 — Deployment, Cost Intelligence & Developer Tooling

- [x] `synapsekit serve` — deploy any pipeline as FastAPI in one command with auto-detection of RAG/Graph/Agent pipelines
- [x] `CostTracker` — hierarchical cost attribution with scope context manager for per-request and per-component tracking
- [x] `BudgetGuard` — budget limits (per-request, per-user, daily) with circuit breaker pattern and automatic enforcement
- [x] `@eval_case` decorator — mark functions as evaluation test cases with pass/fail thresholds
- [x] `synapsekit test` — CLI eval runner with threshold enforcement and JSON/table output formats
- [x] `PromptHub` — local filesystem prompt registry with versioning and retrieval
- [x] `PluginRegistry` — entry-point plugin system for community extensions
- [x] `RedisCheckpointer` + `PostgresCheckpointer` — graph checkpoint backends for distributed and persistent workflows
- [x] 16 providers, 32 tools, 14 loaders, 20 retrieval strategies, 9 memory backends, 1133 tests passing

## v1.3.0 — Cost Routing, Compliance, Audio/Video & More Providers

- [x] `CostRouter` — route to cheapest model meeting quality threshold with `QUALITY_TABLE` + `RouterModelSpec`
- [x] `FallbackChain` — cascade through a list of LLMs on failure with configurable retry logic
- [x] `PIIRedactor` — redact PII (email, phone, SSN, credit card) from text before LLM calls
- [x] `AuditLog` — tamper-evident JSONL audit trail for all agent actions
- [x] `AudioLoader` — transcribe audio files (MP3, WAV, M4A) via Whisper API or local model
- [x] `VideoLoader` — extract and transcribe audio from video files
- [x] `EvalRegressionTracker` — track eval metric changes across versions with pass/fail thresholds
- [x] `MoonshotLLM` — Moonshot AI (Kimi) provider with streaming and function calling
- [x] `ZhipuLLM` — Zhipu AI GLM provider with streaming and function calling
- [x] `CloudflareLLM` — Cloudflare Workers AI provider via native REST API
- [x] 19 providers, 32 tools, 14 loaders, 20 retrieval strategies, 9 memory backends, 1105 tests passing

## v1.4.0 — New Providers, Tools & Multimodal

- [x] `AI21LLM` — AI21 Jamba models (`jamba-1.5-mini`, `jamba-1.5-large`) with 256K context and native function calling
- [x] `DatabricksLLM` — Databricks Foundation Model APIs (DBRX, Llama 3.1, Mixtral) via OpenAI-compatible endpoint
- [x] `ErnieLLM` — Baidu ERNIE Bot (`ernie-4.0`, `ernie-3.5`, `ernie-speed`, `ernie-lite`) for Chinese-English tasks
- [x] `LlamaCppLLM` — local GGUF models via llama-cpp-python with queue+thread true streaming; no API key required
- [x] `APIBuilderTool` — build and execute API calls from OpenAPI specs or natural-language intent; LLM-assisted operation selection
- [x] `GoogleCalendarTool` — create, list, and delete Google Calendar events via Calendar API v3
- [x] `AWSLambdaTool` — invoke AWS Lambda functions with RequestResponse/Event/DryRun types; boto3 credential resolution
- [x] `ImageAnalysisTool` — analyze images with any multimodal LLM; local paths or public URLs; OpenAI + Anthropic format
- [x] `TextToSpeechTool` — text → speech audio via OpenAI TTS; 6 voices, 4 formats
- [x] `SpeechToTextTool` — transcribe audio via Whisper API or local model; delegates to `AudioLoader`
- [x] RAG facade auto-detection extended: `moonshot-*`, `glm-*`, `jamba-*`, `@cf/*`, `@hf/*`, `dbrx-*`/`databricks-*`, `ernie-*`
- [x] 23 providers, 38 tools, 14 loaders, 20 retrieval strategies, 9 memory backends, 1327 tests passing

## v1.4.1 — Community Providers, Tools & Examples (2026-03-27)

- [x] `MinimaxLLM` — Minimax API with SSE streaming; requires `group_id`; auto-detected from `minimax-*` prefix
- [x] `AlephAlphaLLM` — Aleph Alpha Luminous and Pharia models; auto-detected from `luminous-*`/`pharia-*` prefixes
- [x] `YAMLLoader` — load YAML files (list-of-objects or single-object) into Documents; `yaml.safe_load()` based
- [x] `BingSearchTool` — Bing Web Search API v7; auth via `Ocp-Apim-Subscription-Key`
- [x] `WolframAlphaTool` — computational queries via Wolfram Alpha short-answer API
- [x] `examples/` directory — 5 runnable scripts: RAG quickstart, agent tools, graph workflow, multi-provider, caching & retries
- [x] Fixed missing return type annotations in loader helper functions
- [x] 25 providers, 40 tools, 15 loaders, 20 retrieval strategies, 9 memory backends, 1357 tests passing

## v1.5.0 (planned — Q2 2026)

- [ ] **EU Compliance Platform** — `DataResidency` (enforce data never leaves a region), `ConsentManager`, `RetentionPolicy`, `RightToErasure` (GDPR primitives)
- [ ] **EU AI Act risk classifier** — `@risk_level("high")` decorator, risk classification API, automated compliance reports
- [ ] **`EvalCI` GitHub Action** — block deploys on eval metric regressions; cross-version faithfulness/relevancy tracking
- [ ] **Cost analytics dashboard** — optimization recommendations, usage forecasting, per-pipeline cost attribution UI
- [ ] Advanced graph features — time-travel debugging, graph diffing, branch/merge workflows
