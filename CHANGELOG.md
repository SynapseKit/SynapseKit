# Changelog

All notable changes to SynapseKit are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
SynapseKit uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
