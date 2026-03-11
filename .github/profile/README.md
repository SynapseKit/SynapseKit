<div align="center">
  <img src="https://raw.githubusercontent.com/SynapseKit/SynapseKit/main/assets/banner.svg" alt="SynapseKit" width="100%"/>
</div>

<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/synapsekit?color=0a7bbd&label=pypi&logo=pypi&logoColor=white)](https://pypi.org/project/synapsekit/)
[![Python](https://img.shields.io/badge/python-3.14%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-22c55e)](https://github.com/SynapseKit/SynapseKit/blob/main/LICENSE)
[![Tests](https://img.shields.io/badge/tests-332%20passing-22c55e?logo=pytest&logoColor=white)]()

</div>

---

**SynapseKit** is an async-native Python framework for building production-grade LLM applications — RAG pipelines, tool-using agents, and graph workflows — with a clean, minimal API.

Built streaming-first from day one. Two hard dependencies. Every abstraction is transparent, composable, and replaceable.

---

### What's inside

| | |
|---|---|
| **RAG Pipelines** | Streaming retrieval-augmented generation with 5 text splitters, 7+ document loaders, BM25 reranking, conversation memory |
| **Agents** | ReAct and native function calling (OpenAI, Anthropic, Gemini, Mistral). 5 built-in tools, fully extensible |
| **Graph Workflows** | Async DAG pipelines with parallel execution, conditional routing, cycle support, checkpointing, Mermaid export |
| **9 LLM Providers** | OpenAI, Anthropic, Ollama, Gemini, Cohere, Mistral, Bedrock — one interface, swap without rewriting |
| **5 Vector Stores** | InMemory, ChromaDB, FAISS, Qdrant, Pinecone — all behind `VectorStore` ABC |
| **LLM Caching & Retries** | LRU response caching and exponential backoff — opt-in, zero behavior change by default |

### Quick start

```python
from synapsekit import RAG

rag = RAG(model="gpt-4o-mini", api_key="sk-...")
rag.add("Your document text here")

async for token in rag.stream("What is the main topic?"):
    print(token, end="", flush=True)
```

### Links

- [Documentation](https://synapsekit.github.io/synapsekit-docs/)
- [Quickstart](https://synapsekit.github.io/synapsekit-docs/docs/getting-started/quickstart)
- [API Reference](https://synapsekit.github.io/synapsekit-docs/docs/api/llm)
- [Changelog](https://github.com/SynapseKit/SynapseKit/blob/main/CHANGELOG.md)
- [Roadmap](https://synapsekit.github.io/synapsekit-docs/docs/roadmap)

### Install

```bash
pip install synapsekit[openai]    # or [anthropic], [ollama], [all]
```
