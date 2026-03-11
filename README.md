<div align="center">

# SynapseKit

**Async-first Python framework for building LLM applications.**
RAG pipelines · Agents · Graph Workflows · Streaming-native · 2 core dependencies.

[![PyPI version](https://img.shields.io/pypi/v/synapsekit?color=0a7bbd&label=pypi)](https://pypi.org/project/synapsekit/)
[![Python](https://img.shields.io/badge/python-3.14%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-267%20passing-brightgreen)]()
[![Docs](https://img.shields.io/badge/docs-synapsekit.github.io-blue)](https://synapsekit.github.io/synapsekit-docs/)

</div>

---

## What is SynapseKit?

SynapseKit is a Python library for building production-grade LLM applications. It is designed from the ground up to be **async-native** and **streaming-first** — not retrofitted.

- **RAG pipelines** in 3 lines, streaming out of the box
- **Agents** — ReAct loop or native function calling, with built-in tools
- **Graph Workflows** — DAG-based async pipelines with conditional routing and parallel execution
- **9 LLM providers** behind one interface
- **4 vector store backends** behind one interface
- **2 hard dependencies** — `numpy` and `rank-bm25`

---

## Install

```bash
pip install synapsekit[openai]      # OpenAI
pip install synapsekit[anthropic]   # Anthropic
pip install synapsekit[ollama]      # Ollama (local)
pip install synapsekit[gemini]      # Google Gemini
pip install synapsekit[cohere]      # Cohere
pip install synapsekit[mistral]     # Mistral AI
pip install synapsekit[bedrock]     # AWS Bedrock

pip install synapsekit[pdf]         # PDF loader
pip install synapsekit[html]        # HTML loader
pip install synapsekit[web]         # Web (async URL fetch)

pip install synapsekit[chroma]      # ChromaDB
pip install synapsekit[faiss]       # FAISS
pip install synapsekit[qdrant]      # Qdrant
pip install synapsekit[pinecone]    # Pinecone

pip install synapsekit[all]         # Everything
```

---

## RAG

```python
from synapsekit import RAG

rag = RAG(model="gpt-4o-mini", api_key="sk-...")
rag.add("SynapseKit is a Python framework for building LLM applications.")

# Streaming
async for token in rag.stream("What is SynapseKit?"):
    print(token, end="", flush=True)

# Async
answer = await rag.ask("What is SynapseKit?")

# Sync (scripts / notebooks)
answer = rag.ask_sync("What is SynapseKit?")
```

Load documents from any source:

```python
from synapsekit import PDFLoader, CSVLoader, WebLoader, DirectoryLoader

rag.add_documents(PDFLoader("report.pdf").load())
rag.add_documents(CSVLoader("data.csv", text_column="body").load())
rag.add_documents(await WebLoader("https://example.com").load())
rag.add_documents(DirectoryLoader("./docs/").load())  # .txt .pdf .csv .json .html
```

---

## Agents

```python
from synapsekit import AgentExecutor, AgentConfig, CalculatorTool, WebSearchTool
from synapsekit.llm.openai import OpenAILLM
from synapsekit.llm.base import LLMConfig

llm = OpenAILLM(LLMConfig(model="gpt-4o-mini", api_key="sk-..."))

executor = AgentExecutor(AgentConfig(
    llm=llm,
    tools=[CalculatorTool(), WebSearchTool()],
    agent_type="function_calling",  # or "react"
))

answer = await executor.run("What is the square root of 1764, and who invented it?")

# Sync
answer = executor.run_sync("What is 12 factorial?")

# Streaming
async for token in executor.stream("Explain your reasoning step by step"):
    print(token, end="")
```

### Built-in tools

| Tool | Class | Description |
|---|---|---|
| Calculator | `CalculatorTool` | Safe math expression evaluator |
| Python REPL | `PythonREPLTool` | Execute Python with persistent namespace |
| File Read | `FileReadTool` | Read local files |
| Web Search | `WebSearchTool` | DuckDuckGo search (`pip install synapsekit[search]`) |
| SQL Query | `SQLQueryTool` | SQLite / SQLAlchemy SELECT queries |

### Custom tools

```python
from synapsekit import BaseTool, ToolResult

class WeatherTool(BaseTool):
    name = "weather"
    description = "Get current weather for a city"

    async def run(self, city: str) -> ToolResult:
        data = await fetch_weather(city)
        return ToolResult(output=data)
```

---

## Graph Workflows

Build async DAG pipelines — nodes run in waves, parallel nodes execute concurrently.

```python
import asyncio
from synapsekit import StateGraph, END

async def fetch(state):
    return {"data": await api_call(state["query"])}

async def summarise(state):
    return {"summary": await llm_call(state["data"])}

async def classify(state):
    return {"label": "technical" if "code" in state["summary"] else "general"}

def route(state):
    return state["label"]

graph = (
    StateGraph()
    .add_node("fetch", fetch)
    .add_node("summarise", summarise)
    .add_node("classify", classify)
    .add_node("technical_handler", technical_fn)
    .add_node("general_handler", general_fn)
    .add_edge("fetch", "summarise")
    .add_edge("summarise", "classify")
    .add_conditional_edge("classify", route, {
        "technical": "technical_handler",
        "general":   "general_handler",
    })
    .add_edge("technical_handler", END)
    .add_edge("general_handler", END)
    .set_entry_point("fetch")
    .compile()
)

result = asyncio.run(graph.run({"query": "explain async generators"}))
```

Stream node-by-node progress:

```python
async for event in graph.stream({"query": "..."}):
    print(f"[{event['node']}] {event['state']}")
```

Export a Mermaid diagram:

```python
print(graph.get_mermaid())
# flowchart TD
#     __start__ --> fetch
#     fetch --> summarise
#     ...
```

Wrap existing agents or RAG pipelines as graph nodes:

```python
from synapsekit import agent_node, rag_node

graph.add_node("agent", agent_node(executor, input_key="question", output_key="answer"))
graph.add_node("rag",   rag_node(pipeline,  input_key="query",    output_key="answer"))
```

---

## LLM Providers

All providers share one interface. Switch by changing the model name.

```python
from synapsekit import RAG

rag = RAG(model="gpt-4o-mini",                                  api_key="sk-...")           # OpenAI
rag = RAG(model="claude-sonnet-4-6",                            api_key="sk-ant-...")       # Anthropic
rag = RAG(model="gemini-1.5-pro",                               api_key="...", provider="gemini")
rag = RAG(model="command-r-plus",                               api_key="...", provider="cohere")
rag = RAG(model="mistral-large-latest",                         api_key="...", provider="mistral")
rag = RAG(model="llama3",                                       api_key="",   provider="ollama")
rag = RAG(model="anthropic.claude-3-sonnet-20240229-v1:0",      api_key="env", provider="bedrock")
```

Or use any LLM directly:

```python
from synapsekit.llm.openai import OpenAILLM
from synapsekit.llm.base import LLMConfig

llm = OpenAILLM(LLMConfig(model="gpt-4o", api_key="sk-..."))

async for token in llm.stream("Explain transformers in one paragraph"):
    print(token, end="")
```

---

## Vector Stores

```python
from synapsekit import SynapsekitEmbeddings, Retriever, InMemoryVectorStore
from synapsekit.retrieval.chroma import ChromaVectorStore
from synapsekit.retrieval.faiss import FAISSVectorStore
from synapsekit.retrieval.qdrant import QdrantVectorStore

embeddings = SynapsekitEmbeddings()

store = InMemoryVectorStore(embeddings)   # zero-dep, .npz persistence
store = ChromaVectorStore(embeddings)     # pip install synapsekit[chroma]
store = FAISSVectorStore(embeddings)      # pip install synapsekit[faiss]
store = QdrantVectorStore(embeddings)     # pip install synapsekit[qdrant]

# All backends share one interface
retriever = Retriever(store, rerank=True)
await retriever.add(["chunk one", "chunk two", "chunk three"])
results = await store.search("my query", top_k=5)
```

---

## Output Parsers

```python
from synapsekit import JSONParser, ListParser, PydanticParser
from pydantic import BaseModel

# Extract JSON from anywhere in the LLM response
data = JSONParser().parse('Here is the result: {"name": "Alice", "score": 0.95}')

# Parse bullet / numbered lists
items = ListParser().parse("- step one\n- step two\n- step three")

# Validate into a Pydantic model
class Report(BaseModel):
    title: str
    score: float

report = PydanticParser(Report).parse('{"title": "Q1 Analysis", "score": 0.87}')
```

---

## Prompt Templates

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
    examples=[{"input": "2+2", "output": "4"}],
    example_template="Input: {input}\nOutput: {output}",
    suffix="Input: {question}\nOutput:",
).format(question="10 * 7")
```

---

## Observability

```python
rag = RAG(model="gpt-4o-mini", api_key="sk-...", trace=True)
rag.add("Some context")
answer = rag.ask_sync("What is this about?")

summary = rag.tracer.summary()
print(summary["total_tokens"])   # 312
print(summary["total_cost_usd"]) # 0.000087
print(summary["total_calls"])    # 1
```

---

## Development

```bash
git clone https://github.com/SynapseKit/SynapseKit
cd SynapseKit

uv sync --group dev
uv run pytest tests/ -q
```

---

## Documentation

Full docs at **[synapsekit.github.io/synapsekit-docs](https://synapsekit.github.io/synapsekit-docs/)**

---

## License

[MIT](LICENSE)
