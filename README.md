<div align="center">
  <img src="https://raw.githubusercontent.com/SynapseKit/SynapseKit/main/assets/banner.svg" alt="SynapseKit" width="100%"/>
</div>

<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/synapsekit?color=0a7bbd&label=pypi&logo=pypi&logoColor=white)](https://pypi.org/project/synapsekit/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-22c55e)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-795%20passing-22c55e?logo=pytest&logoColor=white)]()
[![Downloads](https://img.shields.io/pypi/dm/synapsekit?color=0a7bbd&logo=pypi&logoColor=white)](https://pypistats.org/packages/synapsekit)
[![Docs](https://img.shields.io/badge/docs-online-0a7bbd?logo=readthedocs&logoColor=white)](https://synapsekit.github.io/synapsekit-docs/)

**[Documentation](https://synapsekit.github.io/synapsekit-docs/) · [Quickstart](https://synapsekit.github.io/synapsekit-docs/docs/getting-started/quickstart) · [API Reference](https://synapsekit.github.io/synapsekit-docs/docs/api/llm) · [Changelog](CHANGELOG.md) · [Report a Bug](https://github.com/SynapseKit/SynapseKit/issues/new?template=bug_report.yml)**

</div>

---

SynapseKit is a Python framework for building production-grade LLM applications. Built **async-native** and **streaming-first** from day one — not retrofitted. Two hard dependencies. Every abstraction is composable, transparent, and replaceable: plain Python you can read, debug, and extend. No magic. No hidden chains. No lock-in.

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
13 LLM providers and 5 vector stores<br/>behind the same API.<br/>Swap without rewriting.
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
- **Teams** who need something they can actually debug and maintain in production

---

## What it covers

<div align="center">

<table>
<tr>
<td width="50%">

**🗂 RAG Pipelines**<br/>
Retrieval-augmented generation with streaming, BM25 reranking, conversation memory, and token tracing. Load from PDFs, URLs, CSVs, HTML, directories, and more.

</td>
<td width="50%">

**🤖 Agents**<br/>
ReAct loop (any LLM) and native function calling (OpenAI / Anthropic / Gemini / Mistral). 19 built-in tools including calculator, Python REPL, web search, SQL, HTTP, shell, summarization, sentiment analysis, and translation. Fully extensible.

</td>
</tr>
<tr>
<td width="50%">

**🔀 Graph Workflows**<br/>
DAG-based async pipelines. Nodes run in waves — parallel nodes execute concurrently. Conditional routing, typed state with reducers, fan-out/fan-in, SSE streaming, event callbacks, human-in-the-loop, checkpointing, and Mermaid export.

</td>
<td width="50%">

**🧠 LLM Providers**<br/>
OpenAI, Anthropic, Ollama, Gemini, Cohere, Mistral, Bedrock, Azure OpenAI, Groq, DeepSeek, OpenRouter, Together, Fireworks — all behind one interface. Auto-detected from the model name. Swap without rewriting.

</td>
</tr>
<tr>
<td width="50%">

**🗄 Vector Stores**<br/>
InMemory (built-in, `.npz` persistence), ChromaDB, FAISS, Qdrant, Pinecone. One interface for all backends.

</td>
<td width="50%">

**🔧 Utilities**<br/>
Output parsers (JSON, Pydantic, List), prompt templates (standard, chat, few-shot), token tracing with cost estimation.

</td>
</tr>
</table>

</div>

---

## Install

**pip**
```bash
pip install synapsekit[openai]       # OpenAI
pip install synapsekit[anthropic]    # Anthropic
pip install synapsekit[ollama]       # Ollama (local)
pip install synapsekit[all]          # Everything
```

**uv**
```bash
uv add synapsekit[openai]
uv add synapsekit[all]
```

**Poetry**
```bash
poetry add synapsekit[openai]
poetry add "synapsekit[all]"
```

Full installation options → [docs](https://synapsekit.github.io/synapsekit-docs/docs/getting-started/installation)

---

## Documentation

Everything you need to get started and go deep is in the docs.

| | |
|---|---|
| 🚀 [Quickstart](https://synapsekit.github.io/synapsekit-docs/docs/getting-started/quickstart) | Up and running in 5 minutes |
| 🗂 [RAG](https://synapsekit.github.io/synapsekit-docs/docs/rag/pipeline) | Pipelines, loaders, retrieval, vector stores |
| 🤖 [Agents](https://synapsekit.github.io/synapsekit-docs/docs/agents/overview) | ReAct, function calling, tools, executor |
| 🔀 [Graph Workflows](https://synapsekit.github.io/synapsekit-docs/docs/graph/overview) | DAG pipelines, conditional routing, parallel execution |
| 🧠 [LLM Providers](https://synapsekit.github.io/synapsekit-docs/docs/llms/overview) | All 13 providers with examples |
| 📖 [API Reference](https://synapsekit.github.io/synapsekit-docs/docs/api/llm) | Full class and method reference |

---

## Development

```bash
git clone https://github.com/SynapseKit/SynapseKit
cd SynapseKit
uv sync --group dev
uv run pytest tests/ -q
```

---

## Contributing

Contributions are welcome — bug reports, documentation fixes, new providers, new features.

Read [CONTRIBUTING.md](CONTRIBUTING.md) to get started. Look for issues tagged [`good first issue`](https://github.com/SynapseKit/SynapseKit/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) if you're new.

---

## Community

- 💬 [Discussions](https://github.com/SynapseKit/SynapseKit/discussions) — ask questions, share ideas
- 🐛 [Bug reports](https://github.com/SynapseKit/SynapseKit/issues/new?template=bug_report.yml)
- 💡 [Feature requests](https://github.com/SynapseKit/SynapseKit/issues/new?template=feature_request.yml)
- 🔒 [Security policy](SECURITY.md)

---

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/AmitoVrito"><img src="https://avatars.githubusercontent.com/u/34062684?v=4" width="100px;" alt="Nautiverse"/><br /><sub><b>Nautiverse</b></sub></a><br /><a href="https://github.com/SynapseKit/SynapseKit/commits?author=AmitoVrito" title="Code">💻</a> <a href="https://github.com/SynapseKit/SynapseKit/commits?author=AmitoVrito" title="Documentation">📖</a> <a href="#maintenance-AmitoVrito" title="Maintenance">🚧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/gordienkoas"><img src="https://avatars.githubusercontent.com/u/127838071?v=4" width="100px;" alt="Gordienko Andrey"/><br /><sub><b>Gordienko Andrey</b></sub></a><br /><a href="https://github.com/SynapseKit/SynapseKit/commits?author=gordienkoas" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Deepak8858"><img src="https://avatars.githubusercontent.com/u/88921480?v=4" width="100px;" alt="Deepak singh"/><br /><sub><b>Deepak singh</b></sub></a><br /><a href="https://github.com/SynapseKit/SynapseKit/commits?author=Deepak8858" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/by22Jy"><img src="https://avatars.githubusercontent.com/u/122969909?v=4" width="100px;" alt="by22Jy"/><br /><sub><b>by22Jy</b></sub></a><br /><a href="https://github.com/SynapseKit/SynapseKit/commits?author=by22Jy" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Arjunkundapur"><img src="https://avatars.githubusercontent.com/u/64265396?v=4" width="100px;" alt="Arjun Kundapur"/><br /><sub><b>Arjun Kundapur</b></sub></a><br /><a href="https://github.com/SynapseKit/SynapseKit/commits?author=Arjunkundapur" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Ashusf90"><img src="https://avatars.githubusercontent.com/u/153393197?v=4" width="100px;" alt="Harshit Gupta"/><br /><sub><b>Harshit Gupta</b></sub></a><br /><a href="https://github.com/SynapseKit/synapsekit-docs/pull/34" title="Documentation">📖</a></td>
    </tr>
  </tbody>
</table>
<!-- ALL-CONTRIBUTORS-LIST:END -->

---

## License

[MIT](LICENSE)
