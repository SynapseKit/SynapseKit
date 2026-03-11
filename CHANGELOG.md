# Changelog

All notable changes to SynapseKit are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
SynapseKit uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
