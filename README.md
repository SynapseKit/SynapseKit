<div align="center">

<h1>SynapseKit</h1>

<p><strong>The async-first Python framework for building LLM applications.</strong><br/>
RAG pipelines · Agents · Graph Workflows · Streaming-native · 2 core dependencies.</p>

[![PyPI version](https://img.shields.io/pypi/v/synapsekit?color=0a7bbd&label=pypi)](https://pypi.org/project/synapsekit/)
[![Python](https://img.shields.io/badge/python-3.14%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-267%20passing-brightgreen)]()
[![Docs](https://img.shields.io/badge/docs-synapsekit.github.io-blue)](https://synapsekit.github.io/synapsekit-docs/)

[Documentation](https://synapsekit.github.io/synapsekit-docs/) · [Quickstart](https://synapsekit.github.io/synapsekit-docs/getting-started/quickstart) · [API Reference](https://synapsekit.github.io/synapsekit-docs/api/llm) · [Report a Bug](https://github.com/SynapseKit/SynapseKit/issues)

</div>

---

## What is SynapseKit?

SynapseKit is a Python framework for building production-grade LLM applications — RAG pipelines, tool-using agents, and async graph workflows — with a clean, minimal API.

It is built async-first and streaming-first from day one, not retrofitted. Every abstraction is designed to be composable, transparent, and easy to replace. There is no magic, no hidden callbacks, no framework lock-in.

---

## Who is it for?

**SynapseKit is for Python developers who want to ship LLM features without fighting their framework.**

- **Backend engineers** adding AI features to existing Python services — async all the way, no event loop surprises
- **ML engineers** building RAG or agent pipelines who need full control over retrieval, prompting, and tool use
- **Researchers and hackers** who want a clean, readable codebase they can actually understand and extend
- **Teams** who have been burned by opaque abstractions and want something they can debug

---

## What problem does it solve?

Building with LLMs in Python today means choosing between:

- Heavy frameworks with hundreds of dependencies, magic abstractions, and APIs that change every release
- Rolling everything yourself — reinventing chunking, retrieval, agent loops, and streaming plumbing

SynapseKit sits in between. It gives you **production-ready building blocks** — loaders, embeddings, vector stores, LLM providers, output parsers, agent loops, graph workflows — that work together out of the box and stay out of your way when you don't need them.

---

## Why SynapseKit?

| | SynapseKit |
|---|---|
| **Async-native** | Every API is `async`/`await` first. Sync wrappers provided for scripts and notebooks. |
| **Streaming-native** | Token-level streaming is the default, not an afterthought. |
| **Minimal dependencies** | 2 hard deps (`numpy`, `rank-bm25`). Everything else is optional. |
| **One interface, many backends** | 9 LLM providers and 4 vector stores behind the same API. Swap without rewriting. |
| **Transparent** | No hidden chains. No magic. Every step is a plain Python function you can read and override. |
| **Composable** | RAG pipeline, agents, and graph nodes are all interchangeable. Wrap a pipeline as an agent tool, or an agent as a graph node. |

---

## Install

```bash
# Pick your LLM provider
pip install synapsekit[openai]      # OpenAI
pip install synapsekit[anthropic]   # Anthropic
pip install synapsekit[ollama]      # Ollama (local)
pip install synapsekit[gemini]      # Google Gemini
pip install synapsekit[cohere]      # Cohere
pip install synapsekit[mistral]     # Mistral AI
pip install synapsekit[bedrock]     # AWS Bedrock

# Document loaders
pip install synapsekit[pdf]         # PDFLoader
pip install synapsekit[html]        # HTMLLoader
pip install synapsekit[web]         # WebLoader (async URL fetch)

# Vector store backends
pip install synapsekit[chroma]      # ChromaDB
pip install synapsekit[faiss]       # FAISS
pip install synapsekit[qdrant]      # Qdrant
pip install synapsekit[pinecone]    # Pinecone

# Everything
pip install synapsekit[all]
```

---

## Quick Start

### RAG in 3 lines

```python
from synapsekit import RAG

rag = RAG(model="gpt-4o-mini", api_key="sk-...")
rag.add("SynapseKit is a Python framework for building LLM applications.")

async for token in rag.stream("What is SynapseKit?"):
    print(token, end="", flush=True)
```

### Agent with tools

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

### Graph workflow

```python
from synapsekit import StateGraph, END

async def fetch(state):   return {"data": await api_call(state["query"])}
async def analyse(state): return {"result": await llm_call(state["data"])}

graph = (
    StateGraph()
    .add_node("fetch", fetch)
    .add_node("analyse", analyse)
    .add_edge("fetch", "analyse")
    .set_entry_point("fetch")
    .set_finish_point("analyse")
    .compile()
)

result = await graph.run({"query": "latest AI research"})
```

---

## Core Features

### RAG Pipelines

Full retrieval-augmented generation with streaming, memory, and token tracing.

```python
from synapsekit import RAG, PDFLoader, DirectoryLoader

rag = RAG(model="gpt-4o-mini", api_key="sk-...", rerank=True, memory_window=10)

# Load from any source
rag.add_documents(PDFLoader("report.pdf").load())
rag.add_documents(DirectoryLoader("./docs/").load())   # .txt .pdf .csv .json .html
rag.add_documents(await WebLoader("https://example.com").load())

# Ask
answer = rag.ask_sync("Summarise the key findings")

# Observe token usage and cost
print(rag.tracer.summary())
# {'total_calls': 1, 'total_tokens': 412, 'total_cost_usd': 0.000062}

# Persist and reload
rag.save("my_index.npz")
rag.load("my_index.npz")
```

### Agents

Two agent strategies, one interface.

**ReAct** (`"react"`) — works with any LLM. Structured Thought → Action → Observation loop.

**Function Calling** (`"function_calling"`) — uses native tool_calls on OpenAI / Anthropic for more reliable tool selection.

```python
executor = AgentExecutor(AgentConfig(
    llm=llm,
    tools=[CalculatorTool(), FileReadTool(), WebSearchTool(), SQLQueryTool()],
    agent_type="function_calling",
    max_iterations=10,
    system_prompt="You are a precise research assistant.",
))

# Async
answer = await executor.run("What is 15% of 48,320?")

# Sync
answer = executor.run_sync("Read ./report.txt and summarise it")

# Streaming
async for token in executor.stream("Explain your steps"):
    print(token, end="")

# Inspect every step
for step in executor.memory.steps:
    print(f"[{step.action}] {step.action_input} → {step.observation}")
```

**Custom tools** — subclass `BaseTool`:

```python
from synapsekit import BaseTool, ToolResult

class WeatherTool(BaseTool):
    name = "weather"
    description = "Get current weather for a city. Input: city name."

    async def run(self, city: str) -> ToolResult:
        data = await fetch_weather_api(city)
        return ToolResult(output=f"{data['temp']}°C, {data['condition']}")
```

### Graph Workflows

Build async DAG pipelines. Nodes run in waves — nodes in the same wave execute concurrently via `asyncio.gather`.

```python
from synapsekit import StateGraph, END

async def ingest(state):   return {"tokens": state["text"].split()}
async def enrich(state):   return {"enriched": annotate(state["tokens"])}
async def store(state):    return {"stored": True}
async def notify(state):   return {"notified": True}

def route(state):
    return "urgent" if state.get("priority") == "high" else "normal"

graph = (
    StateGraph()
    .add_node("ingest", ingest)
    .add_node("enrich", enrich)
    .add_node("store", store)       # these two
    .add_node("notify", notify)     # run in parallel
    .add_edge("ingest", "enrich")
    .add_edge("enrich", "store")
    .add_edge("enrich", "notify")   # fan-out: store + notify run concurrently
    .add_edge("store", END)
    .add_edge("notify", END)
    .set_entry_point("ingest")
    .compile()
)

# Stream node-by-node progress
async for event in graph.stream({"text": "...", "priority": "high"}):
    print(f"✓ {event['node']}")

# Export Mermaid diagram
print(graph.get_mermaid())

# Sync execution
result = graph.run_sync({"text": "..."})
```

Wrap agents or RAG pipelines as graph nodes:

```python
from synapsekit import agent_node, rag_node

graph.add_node("agent", agent_node(executor, input_key="question", output_key="answer"))
graph.add_node("rag",   rag_node(pipeline,   input_key="query",    output_key="context"))
```

### LLM Providers

Nine providers behind one interface. Auto-detected from the model name.

```python
from synapsekit import RAG

rag = RAG(model="gpt-4o-mini",                              api_key="sk-...")
rag = RAG(model="claude-sonnet-4-6",                        api_key="sk-ant-...")
rag = RAG(model="gemini-1.5-pro",                           api_key="...", provider="gemini")
rag = RAG(model="command-r-plus",                           api_key="...", provider="cohere")
rag = RAG(model="mistral-large-latest",                     api_key="...", provider="mistral")
rag = RAG(model="llama3",                                   api_key="",    provider="ollama")
rag = RAG(model="anthropic.claude-3-sonnet-20240229-v1:0",  api_key="env", provider="bedrock")
```

Or use a provider directly for full control:

```python
from synapsekit.llm.openai import OpenAILLM
from synapsekit.llm.base import LLMConfig

llm = OpenAILLM(LLMConfig(model="gpt-4o", api_key="sk-...", temperature=0.0))

async for token in llm.stream("Explain transformers in one paragraph"):
    print(token, end="")
```

### Vector Stores

Four backends, one interface. Swap without rewriting retrieval logic.

```python
from synapsekit import SynapsekitEmbeddings, Retriever, InMemoryVectorStore
from synapsekit.retrieval.chroma  import ChromaVectorStore
from synapsekit.retrieval.faiss   import FAISSVectorStore
from synapsekit.retrieval.qdrant  import QdrantVectorStore

embeddings = SynapsekitEmbeddings(model="all-MiniLM-L6-v2")

store = InMemoryVectorStore(embeddings)   # built-in, .npz persistence
store = ChromaVectorStore(embeddings)
store = FAISSVectorStore(embeddings)
store = QdrantVectorStore(embeddings)

retriever = Retriever(store, rerank=True)  # optional BM25 reranking
await retriever.add(["document one", "document two"])
results = await store.search("query", top_k=5)
```

### Output Parsers

```python
from synapsekit import JSONParser, ListParser, PydanticParser
from pydantic import BaseModel

# Extract JSON from anywhere in the LLM response
data = JSONParser().parse('Sure! Here is the result: {"name": "Alice", "score": 0.95}')

# Parse bullet / numbered lists
items = ListParser().parse("1. Step one\n2. Step two\n3. Step three")

# Validate into a Pydantic model
class Report(BaseModel):
    title: str
    score: float

report = PydanticParser(Report).parse('{"title": "Q1 Analysis", "score": 0.87}')
```

### Prompt Templates

```python
from synapsekit import PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate

# f-string style
prompt = PromptTemplate("Summarise this in {language}: {text}").format(
    language="French", text="..."
)

# Chat messages
messages = ChatPromptTemplate([
    {"role": "system", "content": "You are a {persona}."},
    {"role": "user",   "content": "{question}"},
]).format_messages(persona="data scientist", question="What is overfitting?")

# Few-shot
prompt = FewShotPromptTemplate(
    examples=[{"input": "2+2", "output": "4"}, {"input": "3*3", "output": "9"}],
    example_template="Input: {input}\nOutput: {output}",
    suffix="Input: {question}\nOutput:",
).format(question="10 * 7")
```

---

## Ecosystem

| Component | Description |
|---|---|
| **`synapsekit`** | Core library — RAG, agents, graph workflows, providers |
| **[synapsekit-docs](https://synapsekit.github.io/synapsekit-docs/)** | Full documentation site |

---

## Development

```bash
git clone https://github.com/SynapseKit/SynapseKit
cd SynapseKit

uv sync --group dev        # install dependencies
uv run pytest tests/ -q    # run 267 tests
```

---

## Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like to change.

---

## License

[MIT](LICENSE)
