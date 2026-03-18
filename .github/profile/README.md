<div align="center">

# SynapseKit

### Ship LLM apps faster.

**Production-grade LLM framework for Python.**
Async-native RAG, agents, and graph workflows. 2 dependencies. Zero magic.

<br/>

[![PyPI version](https://img.shields.io/pypi/v/synapsekit?color=0a7bbd&label=pypi&logo=pypi&logoColor=white)](https://pypi.org/project/synapsekit/)
[![Downloads](https://img.shields.io/pypi/dm/synapsekit?color=0a7bbd&logo=pypi&logoColor=white)](https://pypi.org/project/synapsekit/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-22c55e)](https://github.com/SynapseKit/SynapseKit/blob/main/LICENSE)
[![Tests](https://img.shields.io/badge/tests-698%20passing-22c55e?logo=pytest&logoColor=white)](https://github.com/SynapseKit/SynapseKit)
[![GitHub Stars](https://img.shields.io/github/stars/SynapseKit/SynapseKit?style=social)](https://github.com/SynapseKit/SynapseKit)

<br/>

[Documentation](https://synapsekit.github.io/synapsekit-docs/) &bull; [Quickstart](https://synapsekit.github.io/synapsekit-docs/docs/getting-started/quickstart) &bull; [API Reference](https://synapsekit.github.io/synapsekit-docs/docs/api/llm) &bull; [Roadmap](https://synapsekit.github.io/synapsekit-docs/docs/roadmap) &bull; [Contributing](https://github.com/SynapseKit/SynapseKit/blob/main/CONTRIBUTING.md)

</div>

<br/>

---

<br/>

<div align="center">

### Why SynapseKit?

</div>

<table>
<tr>
<td width="50%">

**The problem:** Existing LLM frameworks are heavy — 50+ dependencies, hidden chains, magic callbacks, YAML configs. Hard to debug, harder to ship.

**The fix:** SynapseKit gives you everything you need to build production LLM apps with just **2 core dependencies** and plain Python you can actually read.

</td>
<td width="50%">

```bash
pip install synapsekit[openai]
```

```python
from synapsekit import RAG

rag = RAG(model="gpt-4o-mini", api_key="sk-...")
rag.add("Your document text here")
print(rag.ask_sync("What is the main topic?"))
```

**3 lines. That's it.**

</td>
</tr>
</table>

<br/>

---

<br/>

<div align="center">

### What's inside

</div>

<table>
<tr>
<td align="center" width="33%">
<h4>RAG Pipelines</h4>
5 text splitters &bull; 7+ loaders<br/>
BM25 reranking &bull; conversation memory<br/>
streaming retrieval-augmented generation
</td>
<td align="center" width="33%">
<h4>Agents</h4>
ReAct &bull; native function calling<br/>
OpenAI, Anthropic, Gemini, Mistral<br/>
5 built-in tools &bull; fully extensible
</td>
<td align="center" width="33%">
<h4>Graph Workflows</h4>
parallel execution &bull; conditional routing<br/>
cycle support &bull; checkpointing<br/>
Mermaid export &bull; subgraphs
</td>
</tr>
<tr>
<td align="center">
<h4>9 LLM Providers</h4>
OpenAI &bull; Anthropic &bull; Gemini<br/>
Mistral &bull; Ollama &bull; Cohere<br/>
Bedrock &bull; one interface, swap anytime
</td>
<td align="center">
<h4>5 Vector Stores</h4>
InMemory &bull; ChromaDB &bull; FAISS<br/>
Qdrant &bull; Pinecone<br/>
all behind VectorStore ABC
</td>
<td align="center">
<h4>Production Ready</h4>
LRU response caching<br/>
exponential backoff retries<br/>
332 tests &bull; Apache 2.0 licensed
</td>
</tr>
</table>

<br/>

---

<br/>

<div align="center">

### See it in action

</div>

<table>
<tr>
<td width="50%">

**RAG in 3 lines**

```python
from synapsekit import RAG

rag = RAG(model="gpt-4o-mini", api_key="sk-...")
rag.add("Your document text here")

async for token in rag.stream("What is the main topic?"):
    print(token, end="", flush=True)
```

</td>
<td width="50%">

**Agent with tools**

```python
from synapsekit import FunctionCallingAgent
from synapsekit.agents.tools import CalculatorTool

agent = FunctionCallingAgent(
    llm=llm,
    tools=[CalculatorTool()]
)
result = await agent.run("What is 42 * 17?")
```

</td>
</tr>
<tr>
<td width="50%">

**Graph workflow**

```python
from synapsekit import StateGraph

graph = StateGraph()
graph.add_node("fetch", fetch_data)
graph.add_node("process", process_data)
graph.add_edge("fetch", "process")
graph.set_entry("fetch")
graph.set_finish("process")

app = graph.compile()
result = await app.run({"query": "hello"})
```

</td>
<td width="50%">

**Swap providers in one line**

```python
from synapsekit import RAG

# OpenAI
rag = RAG(model="gpt-4o-mini", api_key="sk-...")

# Anthropic
rag = RAG(model="claude-3-haiku", api_key="sk-ant-...")

# Ollama (local)
rag = RAG(model="ollama/llama3", api_key="")

# Same API. Same code. Different brain.
```

</td>
</tr>
</table>

<br/>

---

<br/>

<div align="center">

### Growing fast

**250+ open issues** &bull; **Contributors welcome** &bull; **Apache 2.0 Licensed**

We're building the most comprehensive async-native LLM framework in Python.
Whether you're a seasoned open-source contributor or looking for your first PR — jump in.

<br/>

[**Star the repo**](https://github.com/SynapseKit/SynapseKit) &bull; [**Browse good first issues**](https://github.com/SynapseKit/SynapseKit/issues?q=label%3A%22good+first+issue%22) &bull; [**Join the discussion**](https://github.com/SynapseKit/SynapseKit/discussions)

<br/>

</div>

---

<div align="center">

[Documentation](https://synapsekit.github.io/synapsekit-docs/) &bull; [PyPI](https://pypi.org/project/synapsekit/) &bull; [Changelog](https://github.com/SynapseKit/SynapseKit/blob/main/CHANGELOG.md) &bull; [Contributing Guide](https://github.com/SynapseKit/SynapseKit/blob/main/CONTRIBUTING.md)

</div>
