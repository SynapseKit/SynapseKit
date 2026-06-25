# Architecture Deep Dive

**SynapseKit v0.9+ â Internal Design Explained**

This document is written for **power users, contributors, and curious engineers** who want to understand how SynapseKit works under the hood.

---

## 1. Async Runtime

SynapseKit is built from the ground up to be **async-first**.

### Core Strategy

```python
# src/synapsekit/_loop.py
def install_fast_loop():
    try:
        import uvloop
        uvloop.install()
    except ImportError:
        pass
```

- Uses `uvloop` when available for maximum performance.
- Falls back gracefully to standard `asyncio`.
- All public APIs are `async` by default (`await graph.run()`, `await agent.run()`, etc.).
- Synchronous wrappers (`ask_sync`, `run_sync`) are provided via `run_sync()` utility for convenience.

```python
# src/synapsekit/_compat.py
T = TypeVar("T")
install_fast_loop()


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)
```

**Diagram**: `docs/architecture/diagrams/async-runtime.html`

---

## 2. Graph Engine (`StateGraph`)

This is the heart of SynapseKit.

### Builder Pattern

```python
graph = StateGraph(state_schema=MyState)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.add_edge("retrieve", "generate")
graph.set_entry_point("retrieve")
compiled = graph.compile()
```

### Compiled Execution Model

The `CompiledGraph` class:
- Builds an adjacency list for O(1) edge lookup
- Executes in **waves** â parallel nodes in the same wave run concurrently using `asyncio.gather()`
- Supports checkpointing, human-in-the-loop (`GraphInterrupt`), and resumability
- Uses a versioned checkpointing system to support schema migrations

**Diagram**: `docs/architecture/diagrams/graph-engine.html`

---

## 3. RAG Facade (`RAG`)

The famous 3-line API is a **facade** over a much richer system.

```python
rag = RAG(model="gpt-4o-mini")
rag.add("my documents...")
answer = rag.ask_sync("What is X?")
```

Under the hood:
- `RAG` class composes `RAGPipeline`, `Retriever`, `Embedder`, `LLM`, and optional `ConversationMemory`
- Uses sensible defaults but is fully configurable
- Supports streaming, evaluation, cost tracking, and guardrails out of the box

```python
# src/synapsekit/rag/pipeline.py
@dataclass
class RAGConfig:
    llm: BaseLLM
    retriever: Retriever
    memory: ConversationMemory
    tracer: TokenTracer | None = None
    retrieval_top_k: int = 5
    system_prompt: str = "Answer using only the provided context."
```

**Diagram**: `docs/architecture/diagrams/rag-facade.html`

---

## 4. Plugin System

SynapseKit has a clean plugin architecture.

```python
class MyPlugin(BasePlugin):
    name = "my_plugin"
    version = "1.0.0"

    async def on_load(self):
        ...
```

- `BasePlugin` provides lifecycle hooks (`on_load`, `on_unload`)
- `PluginLoader` can load plugins from Python files dynamically
- Global registry makes plugins discoverable

```python
# src/synapsekit/plugins/loader.py
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)
```

**Diagram**: `docs/architecture/diagrams/plugin-system.html`

---

## 5. How Everything Fits Together

SynapseKit is designed as a composable stack:

1. **Facade layer** provides simple entry points (`RAG`, `Agent`, `StateGraph`).
2. **Graph engine** orchestrates execution with checkpointing and routing.
3. **LLM + Retrieval + Memory** form the core inference pipeline.
4. **Plugins** extend behavior without altering core code.
5. **Observability** and async runtime underpin all layers.

**Diagram**: `docs/architecture/diagrams/full-architecture.html`

---

## Contributing to the Architecture

If you want to contribute:

- Graph core ? `src/synapsekit/graph/`
- RAG pipeline ? `src/synapsekit/rag/`
- Plugin system ? `src/synapsekit/plugins/`
- Async runtime ? `_loop.py`, `_compat.py`

---

**Status**: Architecture deep-dive complete.
