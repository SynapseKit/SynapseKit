# Changelog

All notable changes to SynapseKit are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
SynapseKit uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
