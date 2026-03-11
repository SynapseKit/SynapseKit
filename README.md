<div align="center">
  <img src="https://raw.githubusercontent.com/SynapseKit/SynapseKit/main/assets/banner.svg" alt="SynapseKit" width="100%"/>
</div>

<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/synapsekit?color=0a7bbd&label=pypi&logo=pypi&logoColor=white)](https://pypi.org/project/synapsekit/)
[![Python](https://img.shields.io/badge/python-3.14%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-22c55e)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-267%20passing-22c55e?logo=pytest&logoColor=white)]()
[![Docs](https://img.shields.io/badge/docs-online-0a7bbd?logo=readthedocs&logoColor=white)](https://synapsekit.github.io/synapsekit-docs/)

**[Documentation](https://synapsekit.github.io/synapsekit-docs/) · [Quickstart](https://synapsekit.github.io/synapsekit-docs/getting-started/quickstart) · [API Reference](https://synapsekit.github.io/synapsekit-docs/api/llm) · [Report a Bug](https://github.com/SynapseKit/SynapseKit/issues)**

</div>

---

SynapseKit is a Python framework for building production-grade LLM applications. It is **async-native** and **streaming-first** by design — not retrofitted. Every abstraction is composable, transparent, and replaceable. No magic. No hidden callbacks. No lock-in.

---

<div align="center">

<table>
<tr>
<td align="center" width="33%">
<h3>⚡ Async-native</h3>
Every API is <code>async/await</code> first.<br/>Sync wrappers for scripts and notebooks.<br/>No event loop surprises.
</td>
<td align="center" width="33%">
<h3>🌊 Streaming-first</h3>
Token-level streaming is the default,<br/>not an afterthought.<br/>Works across all providers.
</td>
<td align="center" width="33%">
<h3>🪶 Minimal footprint</h3>
2 hard dependencies: <code>numpy</code> + <code>rank-bm25</code>.<br/>Everything else is optional.<br/>Install only what you use.
</td>
</tr>
<tr>
<td align="center" width="33%">
<h3>🔌 One interface</h3>
9 LLM providers and 4 vector stores<br/>behind the same API.<br/>Swap without rewriting.
</td>
<td align="center" width="33%">
<h3>🧩 Composable</h3>
RAG pipelines, agents, and graph nodes<br/>are interchangeable.<br/>Wrap anything as anything.
</td>
<td align="center" width="33%">
<h3>🔍 Transparent</h3>
No hidden chains.<br/>Every step is plain Python<br/>you can read and override.
</td>
</tr>
</table>

</div>

---

## Who is it for?

SynapseKit is for Python developers who want to ship LLM features without fighting their framework.

- **Backend engineers** adding AI features to existing Python services
- **ML engineers** building RAG or agent pipelines who need full control over retrieval, prompting, and tool use
- **Researchers and hackers** who want a clean, readable codebase they can understand and extend
- **Teams** who need something they can actually debug and maintain

---

## Install

```bash
pip install synapsekit[openai]       # OpenAI
pip install synapsekit[anthropic]    # Anthropic
pip install synapsekit[ollama]       # Ollama (local)
pip install synapsekit[gemini]       # Google Gemini
pip install synapsekit[cohere]       # Cohere
pip install synapsekit[mistral]      # Mistral AI
pip install synapsekit[bedrock]      # AWS Bedrock
pip install synapsekit[all]          # Everything
```

---

## Quick Start

<details open>
<summary><strong>RAG — 3 lines</strong></summary>

```python
from synapsekit import RAG

rag = RAG(model="gpt-4o-mini", api_key="sk-...")
rag.add("SynapseKit is a Python framework for building LLM applications.")

async for token in rag.stream("What is SynapseKit?"):
    print(token, end="", flush=True)
```

</details>

<details>
<summary><strong>Agent with tools</strong></summary>

```python
from synapsekit import AgentExecutor, AgentConfig, CalculatorTool, WebSearchTool
from synapsekit.llm.openai import OpenAILLM
from synapsekit.llm.base import LLMConfig

llm = OpenAILLM(LLMConfig(model="gpt-4o-mini", api_key="sk-..."))

executor = AgentExecutor(AgentConfig(
    llm=llm,
    tools=[CalculatorTool(), WebSearchTool()],
    agent_type="function_calling",
))

answer = await executor.run("What is the square root of 1764?")
```

</details>

<details>
<summary><strong>Graph workflow</strong></summary>

```python
from synapsekit import StateGraph, END

async def fetch(state):    return {"data": await api_call(state["query"])}
async def summarise(state): return {"result": await llm_call(state["data"])}

graph = (
    StateGraph()
    .add_node("fetch", fetch)
    .add_node("summarise", summarise)
    .add_edge("fetch", "summarise")
    .set_entry_point("fetch")
    .set_finish_point("summarise")
    .compile()
)

result = await graph.run({"query": "latest AI research"})
```

</details>

---

## RAG

Full retrieval-augmented generation with streaming, BM25 reranking, conversation memory, and token tracing.

```python
from synapsekit import RAG, PDFLoader, DirectoryLoader

rag = RAG(model="gpt-4o-mini", api_key="sk-...", rerank=True, memory_window=10)

# Load from any source
rag.add_documents(PDFLoader("report.pdf").load())
rag.add_documents(DirectoryLoader("./docs/").load())   # .txt .pdf .csv .json .html
rag.add_documents(await WebLoader("https://example.com").load())

# Ask
answer = rag.ask_sync("Summarise the key findings")

# Token usage and cost
print(rag.tracer.summary())
# {'total_calls': 1, 'total_tokens': 412, 'total_cost_usd': 0.000062}

# Persist and reload
rag.save("my_index.npz")
rag.load("my_index.npz")
```

---

## Agents

Two strategies, one interface. Both support `run()`, `stream()`, and `run_sync()`.

| Strategy | Class | Best for |
|---|---|---|
| **ReAct** | `ReActAgent` | Any LLM — structured Thought → Action → Observation loop |
| **Function Calling** | `FunctionCallingAgent` | OpenAI / Anthropic — native tool_calls for reliable selection |

```python
executor = AgentExecutor(AgentConfig(
    llm=llm,
    tools=[CalculatorTool(), FileReadTool(), WebSearchTool(), SQLQueryTool()],
    agent_type="function_calling",
    max_iterations=10,
))

answer = await executor.run("What is 15% of 48,320?")
answer = executor.run_sync("Read ./report.txt and summarise it")

async for token in executor.stream("Explain your reasoning"):
    print(token, end="")
```

**Built-in tools:**

| Tool | Description |
|---|---|
| `CalculatorTool` | Safe math expression evaluator |
| `PythonREPLTool` | Execute Python with persistent namespace |
| `FileReadTool` | Read local files |
| `WebSearchTool` | DuckDuckGo search (`pip install synapsekit[search]`) |
| `SQLQueryTool` | SQLite / SQLAlchemy SELECT queries |

**Custom tools** — one class, one method:

```python
from synapsekit import BaseTool, ToolResult

class WeatherTool(BaseTool):
    name = "weather"
    description = "Get current weather for a city. Input: city name."

    async def run(self, city: str) -> ToolResult:
        data = await fetch_weather_api(city)
        return ToolResult(output=f"{data['temp']}°C, {data['condition']}")
```

---

## Graph Workflows

Build async DAG pipelines. Nodes in the same wave run concurrently via `asyncio.gather`. Conditional routing at runtime. Compile-time cycle detection.

```python
from synapsekit import StateGraph, END

graph = (
    StateGraph()
    .add_node("ingest",   ingest_fn)
    .add_node("enrich",   enrich_fn)
    .add_node("store",    store_fn)    # ─┐ these two nodes
    .add_node("notify",   notify_fn)   # ─┘ run in parallel
    .add_edge("ingest", "enrich")
    .add_edge("enrich", "store")       # fan-out:
    .add_edge("enrich", "notify")      # store + notify run concurrently
    .add_edge("store",  END)
    .add_edge("notify", END)
    .set_entry_point("ingest")
    .compile()
)
```

**Conditional routing:**

```python
def route(state):
    return "urgent" if state["priority"] == "high" else "normal"

graph.add_conditional_edge("classify", route, {
    "urgent": "fast_handler",
    "normal": "slow_handler",
})
```

**Stream node-by-node progress:**

```python
async for event in graph.stream({"query": "..."}):
    print(f"✓ {event['node']}")
```

**Mermaid diagram export:**

```python
print(graph.get_mermaid())
# flowchart TD
#     __start__ --> ingest
#     ingest --> enrich
#     enrich --> store
#     enrich --> notify
#     ...
```

**Wrap agents or RAG pipelines as nodes:**

```python
from synapsekit import agent_node, rag_node

graph.add_node("agent", agent_node(executor, input_key="question", output_key="answer"))
graph.add_node("rag",   rag_node(pipeline,   input_key="query",    output_key="context"))
```

---

## LLM Providers

Nine providers behind one interface. Auto-detected from the model name.

```python
from synapsekit import RAG

RAG(model="gpt-4o-mini",                             api_key="sk-...")
RAG(model="claude-sonnet-4-6",                       api_key="sk-ant-...")
RAG(model="gemini-1.5-pro",                          api_key="...",  provider="gemini")
RAG(model="command-r-plus",                          api_key="...",  provider="cohere")
RAG(model="mistral-large-latest",                    api_key="...",  provider="mistral")
RAG(model="llama3",                                  api_key="",     provider="ollama")
RAG(model="anthropic.claude-3-sonnet-20240229-v1:0", api_key="env",  provider="bedrock")
```

---

## Vector Stores

Four backends, one interface.

```python
from synapsekit import InMemoryVectorStore, SynapsekitEmbeddings, Retriever
from synapsekit.retrieval.chroma import ChromaVectorStore    # pip install synapsekit[chroma]
from synapsekit.retrieval.faiss  import FAISSVectorStore     # pip install synapsekit[faiss]
from synapsekit.retrieval.qdrant import QdrantVectorStore    # pip install synapsekit[qdrant]

embeddings = SynapsekitEmbeddings()

retriever = Retriever(InMemoryVectorStore(embeddings), rerank=True)
await retriever.add(["chunk one", "chunk two", "chunk three"])
results = await retriever.retrieve("my query", top_k=5)
```

---

## Development

```bash
git clone https://github.com/SynapseKit/SynapseKit
cd SynapseKit

uv sync --group dev
uv run pytest tests/ -q
# 267 passed, 6 skipped
```

---

## Documentation

Full docs at **[synapsekit.github.io/synapsekit-docs](https://synapsekit.github.io/synapsekit-docs/)**

---

## License

[MIT](LICENSE)
