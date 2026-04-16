<div align="center">
  <img src="https://raw.githubusercontent.com/SynapseKit/SynapseKit/main/assets/banner.svg" alt="SynapseKit" width="100%"/>
</div>

<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/synapsekit?color=22c55e&label=pypi&logo=pypi&logoColor=white)](https://pypi.org/project/synapsekit/)
[![Python](https://img.shields.io/badge/python-3.10%2B-22c55e?logo=python&logoColor=white)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-22c55e)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-2041%20passing-22c55e?logo=pytest&logoColor=white)]()
[![Downloads](https://img.shields.io/pypi/dm/synapsekit?color=22c55e&logo=pypi&logoColor=white)](https://pypistats.org/packages/synapsekit)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/synapsekit?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/synapsekit)
[![Docs](https://img.shields.io/badge/docs-online-22c55e?logo=readthedocs&logoColor=white)](https://synapsekit.github.io/synapsekit-docs/)
[![Discord](https://img.shields.io/discord/1488136255597182988?logo=discord&logoColor=white)](https://discord.gg/PSuAXHRywJ)

**[Documentation](https://synapsekit.github.io/synapsekit-docs/) · [Quickstart](https://synapsekit.github.io/synapsekit-docs/docs/getting-started/quickstart) · [API Reference](https://synapsekit.github.io/synapsekit-docs/docs/api/llm) · [Changelog](CHANGELOG.md) · [Discord](https://discord.gg/PSuAXHRywJ) · [Report a Bug](https://github.com/SynapseKit/SynapseKit/issues/new?template=bug_report.yml)**

</div>

---

**Build production LLM apps with 2 dependencies.**
Async-native RAG, Agents, and Graph workflows — no magic, no SaaS, no bloat.

> *"LangChain for people who hate LangChain."*

SynapseKit is the minimal, async-first Python framework for LLM applications. 33 providers · 48 tools · 46 loaders · 10 vector stores. Every abstraction is plain Python you can read, debug, and extend. No hidden chains. No global state. No lock-in.

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
31 LLM providers and 9 vector stores<br/>behind the same API.<br/>Swap without rewriting.
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

## SynapseKit vs LangChain vs LlamaIndex

<div align="center">

| | SynapseKit | LangChain | LlamaIndex |
|---|---|---|---|
| Hard dependencies | **2** | 50+ | 20+ |
| Install size | **~5 MB** | ~200 MB+ | ~100 MB+ |
| Async-native | **✅ Default** | ⚠️ Partial | ⚠️ Partial |
| Cost tracking | **✅ Built-in** | ❌ LangSmith (SaaS) | ❌ No |
| Evaluation | **✅ CLI + GitHub Action** | ❌ LangSmith (SaaS) | ✅ Built-in |
| Graph workflows | **✅ Built-in** | ✅ LangGraph (separate pkg) | ❌ No |
| LLM providers | **31** | 38+ | 20+ |
| Stack traces | **Your code** | Framework internals | Framework internals |

</div>

LangChain has more raw integrations and more tutorials. That's not what SynapseKit is optimizing for. SynapseKit is optimizing for the engineer who needs to ship, debug, and maintain an LLM feature in production — where readable code, predictable async behavior, and no surprise SaaS bills actually matter.

---

## Who is it for?

SynapseKit is for Python developers who want to ship LLM features without fighting their framework.

- **Burned LangChain users** — hit a wall with debugging, dependency hell, or version churn and want full control back
- **Async backend engineers** — building FastAPI services where LangChain's sync-first model feels bolted on
- **Cost-conscious teams** — startups and teams who don't want a LangSmith subscription for basic observability
- **ML engineers** — building RAG or agent pipelines who need full control over retrieval, prompting, and tool use

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
ReAct loop (any LLM) and native function calling (OpenAI / Anthropic / Gemini / Mistral). 48 built-in tools including calculator, Python REPL, code interpreter, web search, SQL, HTTP, shell, Twilio, arxiv, pubmed, wolfram, wikipedia, and more. Fully extensible.

</td>
</tr>
<tr>
<td width="50%">

**🔀 Graph Workflows**<br/>
DAG-based async pipelines. Nodes run in waves — parallel nodes execute concurrently. Conditional routing, typed state with reducers, fan-out/fan-in, SSE streaming, event callbacks, human-in-the-loop, checkpointing, and Mermaid export.

</td>
<td width="50%">

**🧠 LLM Providers**<br/>
OpenAI, Anthropic, Ollama, Gemini, Cohere, Mistral, Bedrock, Azure OpenAI, Groq, DeepSeek, OpenRouter, Together, Fireworks, Cerebras, Cloudflare, Moonshot, Perplexity, Vertex AI, Zhipu, AI21 Labs, Databricks, Baidu ERNIE, llama.cpp, LM Studio, Minimax, Aleph Alpha, Hugging Face, SambaNova, xAI, NovitaAI, Writer — all behind one interface. Auto-detected from the model name. Swap without rewriting.

</td>
</tr>
<tr>
<td width="50%">

**🗄 Vector Stores**<br/>
InMemory (built-in, `.npz` persistence), ChromaDB, FAISS, Qdrant, Pinecone, Weaviate, PGVector, Milvus, LanceDB. One interface for all 9 backends.

</td>
<td width="50%">

**🔧 Utilities**<br/>
Output parsers (JSON, Pydantic, List), prompt templates (standard, chat, few-shot), token tracing with cost estimation.

</td>
</tr>
<tr>
<td width="50%" colspan="2">

**🧪 EvalCI — LLM Quality Gates**<br/>
GitHub Action that runs `@eval_case` suites on every PR and blocks merge if quality drops. No infrastructure, 2-minute setup. Score, cost, and latency tracked per case. Works with any LLM provider. → [GitHub Marketplace](https://github.com/marketplace/actions/evalci-by-synapsekit) · [Docs](https://synapsekit.github.io/synapsekit-docs/docs/evalci/overview)

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
| 🧠 [LLM Providers](https://synapsekit.github.io/synapsekit-docs/docs/llms/overview) | All 33 providers with examples |
| 🧪 [EvalCI](https://synapsekit.github.io/synapsekit-docs/docs/evalci/overview) | LLM quality gates on every PR — GitHub Action |
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

- 💬 [Discord](https://discord.gg/PSuAXHRywJ) — chat, help, show and tell
- 💬 [Discussions](https://github.com/SynapseKit/SynapseKit/discussions) — ask questions, share ideas
- 🧭 [Discord roles draft](DISCORD_ROLES.md) — proposed roles and permissions for issue #389
- 🧭 [Discord release webhook draft](DISCORD_RELEASE_WEBHOOKS.md) — automate release announcements for issue #390
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
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/DhruvGarg111"><img src="https://avatars.githubusercontent.com/u/136477030?v=4" width="100px;" alt="Dhruv Garg"/><br /><sub><b>Dhruv Garg</b></sub></a><br /><a href="https://github.com/SynapseKit/SynapseKit/commits?author=DhruvGarg111" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/adaumsilva"><img src="https://avatars.githubusercontent.com/u/178027480?v=4" width="100px;" alt="Adam Silva"/><br /><sub><b>Adam Silva</b></sub></a><br /><a href="https://github.com/SynapseKit/SynapseKit/commits?author=adaumsilva" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/qorexdev"><img src="https://avatars.githubusercontent.com/u/248982649?v=4" width="100px;" alt="qorex"/><br /><sub><b>qorex</b></sub></a><br /><a href="https://github.com/SynapseKit/SynapseKit/commits?author=qorexdev" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Abhay-Mmmm"><img src="https://avatars.githubusercontent.com/u/192120538?v=4" width="100px;" alt="Abhay Krishna"/><br /><sub><b>Abhay Krishna</b></sub></a><br /><a href="https://github.com/SynapseKit/SynapseKit/commits?author=Abhay-Mmmm" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ayushbhatt1224"><img src="https://avatars.githubusercontent.com/u/129763284?v=4" width="100px;" alt="AYUSH BHATT"/><br /><sub><b>AYUSH BHATT</b></sub></a><br /><a href="https://github.com/SynapseKit/SynapseKit/commits?author=ayushbhatt1224" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Chaturvediharsh123"><img src="https://avatars.githubusercontent.com/u/146837343?v=4" width="100px;" alt="HARSH"/><br /><sub><b>HARSH</b></sub></a><br /><a href="https://github.com/SynapseKit/SynapseKit/commits?author=Chaturvediharsh123" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/mikemolinet"><img src="https://avatars.githubusercontent.com/u/237856306?v=4" width="100px;" alt="mikemolinet"/><br /><sub><b>mikemolinet</b></sub></a><br /><a href="https://github.com/SynapseKit/SynapseKit/commits?author=mikemolinet" title="Code">💻</a> <a href="https://github.com/SynapseKit/SynapseKit/issues?q=author%3Amikemolinet" title="Bug reports">🐛</a></td>
    </tr>
  </tbody>
</table>
<!-- ALL-CONTRIBUTORS-LIST:END -->

---

## License

[Apache 2.0](LICENSE)
