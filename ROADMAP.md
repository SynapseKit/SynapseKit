# SynapseKit Roadmap

## Vision
Beat LangChain. Start by owning the async-native Python RAG niche, then expand to cover everything LangChain offers — but with a simpler, faster, more Pythonic API.

Positioning: "What FastAPI did to Flask/Django, SynapseKit does to LangChain."

---

## Design Principles (never break these)
1. `stream()` is always primary — `generate()` is just `"".join([...async for...])`
2. All I/O is `async`. Sync wrappers use `asyncio.run()`.
3. Every external import is lazy with clear error messages
4. No global state — every class is independently instantiable
5. Hard deps: `numpy` + `rank-bm25` only. Everything else optional.
6. No chains, no callbacks, no magic — just async functions and plain classes

---

## ✅ Phase 1 — Core RAG (DONE)
- [x] `BaseLLM` ABC + `LLMConfig`
- [x] `OpenAILLM` — async streaming
- [x] `AnthropicLLM` — async streaming
- [x] `SynapsekitEmbeddings` — sentence-transformers backend (no proprietary deps)
- [x] `InMemoryVectorStore` — numpy cosine sim + .npz persistence
- [x] `Retriever` — vector search + optional BM25 rerank (rank-bm25)
- [x] `TextSplitter` — pure Python recursive character splitter, zero deps
- [x] `ConversationMemory` — sliding window
- [x] `TokenTracer` — tokens, latency, cost per call
- [x] `TextLoader`, `StringLoader`
- [x] `RAGPipeline` — full orchestrator
- [x] `RAG` facade — 3-line happy path
- [x] `run_sync()` — works inside/outside event loops
- [x] 52 tests, all passing
- [x] Zero proprietary dependencies (removed chunkrank)

---

## 🔜 Phase 2 — Own the Niche
**Goal: be the undisputed best async RAG library. Make LangChain's RAG story look bad.**

### More loaders
- [ ] `PDFLoader` (pypdf)
- [ ] `HTMLLoader` (beautifulsoup4)
- [ ] `CSVLoader`
- [ ] `JSONLoader`
- [ ] `DirectoryLoader` — load all files in a folder
- [ ] `WebLoader` — fetch and parse a URL

### Output parsers
- [ ] `JSONParser` — extract JSON from LLM output
- [ ] `PydanticParser` — parse into a Pydantic model
- [ ] `ListParser` — bullet/numbered list → Python list

### More vector stores (pluggable ABC)
- [ ] `VectorStore` ABC — swap any backend
- [ ] `ChromaVectorStore`
- [ ] `FAISSVectorStore`
- [ ] `QdrantVectorStore`
- [ ] `PineconeVectorStore`

### More LLM providers
- [ ] `OllamaLLM` — local models
- [ ] `CohereLLM`
- [ ] `MistralLLM`
- [ ] `GeminiLLM` (Google)
- [ ] `BedrockLLM` (AWS)

### Prompt templates
- [ ] `PromptTemplate` — f-string style, no magic
- [ ] `ChatPromptTemplate` — message list builder
- [ ] `FewShotPromptTemplate`

### Benchmark suite (publish results publicly)
- [ ] Simplicity: lines of code vs LangChain/LlamaIndex for same task
- [ ] Performance: time to first token, p50/p95/p99 latency
- [ ] Throughput: concurrent requests
- [ ] Install footprint: size (MB), dep count, cold import time
- [ ] RAG quality: HotpotQA / TriviaQA / MS MARCO (faithfulness, relevancy)

---

## 🔜 Phase 3 — Agents
**This is where you get taken seriously as a LangChain alternative.**

- [ ] `BaseTool` ABC — `name`, `description`, `run(input) -> str`
- [ ] Tool registry — tools by name
- [ ] `ReActAgent` — Reasoning + Acting loop
- [ ] `FunctionCallingAgent` — native OpenAI/Anthropic tool use
- [ ] `AgentExecutor` — loop runner, max iterations, error recovery
- [ ] `AgentMemory` — agent scratchpad
- [ ] Built-in tools:
  - [ ] `WebSearchTool` (Brave / Serper API)
  - [ ] `CalculatorTool`
  - [ ] `PythonREPLTool`
  - [ ] `FileReadTool`
  - [ ] `SQLQueryTool`

### Agent benchmarks
- [ ] ToolBench / AgentBench task success rate
- [ ] Tool call accuracy vs LangChain agents
- [ ] Avg steps to task completion

---

## 🔜 Phase 4 — Graph Workflows (LangGraph killer)
**LangChain is doubling down on LangGraph. We need a graph primitive.**

- [ ] `Graph` — DAG-based workflow primitive
- [ ] `Node` — wraps any async callable
- [ ] `Edge`, `ConditionalEdge` — routing between nodes
- [ ] Parallel node execution (async-native advantage)
- [ ] `GraphExecutor` — runs the DAG
- [ ] Mermaid diagram export
- [ ] State management across graph steps

---

## 🔜 Phase 5 — Structured Output + Evaluation
- [ ] Force LLM to return a Pydantic model (JSON mode + validation)
- [ ] Retry on parse failure with error feedback to LLM
- [ ] `Evaluator` — answer faithfulness, relevancy, groundedness
- [ ] Built-in RAGAS-style metrics
- [ ] Retrieval precision@k, recall@k
- [ ] Benchmark runner CLI: `synapsekit benchmark --vs langchain`

---

## 🔜 Phase 6 — Platform (where the money is)
- [ ] **Local observability UI** — full trace viewer, span replay, call graph (LangSmith-style, fully local, open source)
- [ ] **Streaming UI helpers** — SSE + WebSocket adapters for FastAPI
- [ ] **Multi-modal** — images in prompts (OpenAI vision, Claude)
- [ ] **`synapsekit serve`** — deploy any RAG/agent app as FastAPI in one command
- [ ] **Prompt hub** — versioned prompt registry

---

## Competitor Coverage Matrix

| Feature | LangChain | LlamaIndex | SynapseKit Target |
|---------|-----------|------------|----------------|
| LLM providers | 50+ | 40+ | 10+ (Phase 2) |
| Vector stores | 20+ | 20+ | 8+ (Phase 2) |
| Document loaders | 100+ | 80+ | 20+ (Phase 2) |
| Agents | ✓ | ✓ | Phase 3 |
| Graph workflows | LangGraph | ✓ | Phase 4 |
| Structured output | ✓ | ✓ | Phase 5 |
| Evaluation | ✓ | ✓ | Phase 5 |
| Observability | LangSmith (paid) | ✓ | Phase 6 (free) |
| Streaming-native | ✗ | ✗ | ✓ NOW |
| Async-native | Partial | Partial | ✓ NOW |
| Install size | ~500MB | ~400MB | ~50MB NOW |
| Zero magic | ✗ | ✗ | ✓ NOW |
