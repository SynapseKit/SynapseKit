<div align="center">
  <img src="https://raw.githubusercontent.com/SynapseKit/SynapseKit/main/assets/banner.svg" alt="SynapseKit" width="100%"/>
</div>

<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/synapsekit?color=22c55e&label=pypi&logo=pypi&logoColor=white)](https://pypi.org/project/synapsekit/)
[![Python](https://img.shields.io/badge/python-3.10%2B-22c55e?logo=python&logoColor=white)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-22c55e)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-3600%20passing-22c55e?logo=pytest&logoColor=white)]()
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

SynapseKit is the minimal, async-first Python framework for LLM applications. 33 providers · 48+ tools · 64 loaders · 22 vector stores. Every abstraction is plain Python you can read, debug, and extend. No hidden chains. No global state. No lock-in.

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
33 LLM providers and 22 vector stores<br/>behind the same API.<br/>Swap without rewriting.
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

## 10-Line Agent Example

```python
from synapsekit import agent, tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 22°C in {city}"

my_agent = agent(
    model="gpt-4o-mini",
    api_key="sk-...",
    tools=[get_weather],
)

print(my_agent.run("What's the weather in Tokyo?"))
```

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
| LLM providers | **33** | 38+ | 20+ |
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
InMemory (built-in, `.npz` persistence), ChromaDB, FAISS, Qdrant, Pinecone, Weaviate, PGVector, Milvus, LanceDB, SQLiteVec, MongoDB Atlas, Redis, Elasticsearch, OpenSearch, Supabase, Cassandra, DuckDB, ClickHouse, Marqo, Typesense, Vespa, Zilliz. One interface for all 22 backends.

</td>
<td width="50%">

**🔧 Utilities**<br/>
Output parsers (JSON, Pydantic, List), prompt templates (standard, chat, few-shot), token tracing with cost estimation.

</td>
</tr>
<tr>
<td width="50%">

**🧠 Reasoning LLMs** *(new in v1.7.0)*<br/>
`ReasoningLLM` unified adapter for o1/o3, Claude thinking, Gemini thinking, DeepSeek R1, and Qwen QwQ. Returns `ReasoningResponse` with answer, thinking trace, and token breakdown. `stream()` yields `ReasoningStreamChunk` with `is_thinking` flag.

</td>
<td width="50%">

**⚖️ Cost-Quality Routing** *(new in v1.7.0)*<br/>
`CostQualityRouter` explores candidates round-robin then exploits the cheapest model meeting your quality threshold. Tracks Pareto frontier of cost vs quality. Optional `budget_per_call_usd` hard cap.

</td>
</tr>
<tr>
<td width="50%">

**🎯 Prompt Optimization** *(new in v1.7.0)*<br/>
`PromptOptimizer` scores prompt variants against an `@eval_case` suite and returns the best `PromptCandidate`. Supports LLM-generated variants or manual lists. Budget-aware early stopping.

</td>
<td width="50%">

**🌐 Federated Retrieval** *(new in v1.7.0)*<br/>
`FederatedRetriever` fans out to multiple local retrievers and remote HTTP endpoints in parallel. RRF, normalised score fusion, or round-robin interleave. Near-duplicate dedup, per-source timeouts.

</td>
</tr>
<tr>
<td width="50%" colspan="2">

**⚡ Performance suite** *(new in v1.7.0)*<br/>
`orjson` fast JSON across all hot paths · `uvloop` event loop · `xxhash` cache key hashing (5–10× faster) · pre-allocated vector buffer (O(1) amortised inserts) · vectorised MMR · `__slots__` on hot classes · optional Rust extension for chunking and hashing. Install with `pip install synapsekit[performance]`.

</td>
</tr>
<tr>
<td width="50%" colspan="2">

**🧪 EvalCI — LLM Quality Gates**<br/>
GitHub Action that runs `@eval_case` suites on every PR and blocks merge if quality drops. No infrastructure, 2-minute setup. Score, cost, and latency tracked per case. Works with any LLM provider. → [GitHub Marketplace](https://github.com/marketplace/actions/evalci-by-synapsekit) · [Docs](https://synapsekit.github.io/synapsekit-docs/docs/evalci/overview)

</td>
</tr>
<tr>
<td width="50%" colspan="2">

**📊 Agent Benchmarking**<br/>
Evaluate your agents against industry-standard benchmarks like GAIA, SWE-bench, WebArena, and AgentBench directly from the CLI. Generate leaderboards to compare performance across tasks.

**🧪 EvalHub Community Suites**<br/>
Run shared community eval suites with `synapsekit bench` and compare aggregate score against baseline.
</td>
</tr>
</table>

</div>

### EvalHub quick usage

```bash
synapsekit bench --list
synapsekit bench --suite community/customer-support --model gpt-4o-mini
synapsekit bench --publish my_evals/ --name myorg/rag-finance
```

Docs: [docs/evalhub.md](docs/evalhub.md)

---

## Integrations

<div align="center">

### One interface. 190+ integrations. Zero lock-in.

| 🧠 LLM Providers | 🗄 Vector Stores | 📂 Data Loaders | 🔧 Agent Tools |
|:---:|:---:|:---:|:---:|
| **33** | **22** | **64** | **48+** |

Every integration is `pip install synapsekit[name]` — nothing else. Swap providers, vector stores, or loaders without touching your application code.

</div>

> Icons use [Google Favicons](https://google.com/s2/favicons) for reliability across light and dark themes.

### 🧠 LLM Providers — 33 supported

> Every provider implements the same `BaseLLM` interface. Auto-detected from model name — `gpt-4o` → OpenAI, `claude-*` → Anthropic, `gemini-*` → Google. **Swap without rewriting.**

<table>
  <tr>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=openai.com&sz=128" height="40" alt="OpenAI"/><br/><sub><b>OpenAI</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=anthropic.com&sz=128" height="40" alt="Anthropic"/><br/><sub><b>Anthropic</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=gemini.google.com&sz=128" height="40" alt="Google Gemini"/><br/><sub><b>Gemini</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=azure.microsoft.com&sz=128" height="40" alt="Azure OpenAI"/><br/><sub><b>Azure OpenAI</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=aws.amazon.com&sz=128" height="40" alt="AWS Bedrock"/><br/><sub><b>AWS Bedrock</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=cloud.google.com&sz=128" height="40" alt="Vertex AI"/><br/><sub><b>Vertex AI</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=mistral.ai&sz=128" height="40" alt="Mistral"/><br/><sub><b>Mistral</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=cohere.com&sz=128" height="40" alt="Cohere"/><br/><sub><b>Cohere</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=groq.com&sz=128" height="40" alt="Groq"/><br/><sub><b>Groq</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=huggingface.co&sz=128" height="40" alt="Hugging Face"/><br/><sub><b>Hugging Face</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=cloudflare.com&sz=128" height="40" alt="Cloudflare"/><br/><sub><b>Cloudflare</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=databricks.com&sz=128" height="40" alt="Databricks"/><br/><sub><b>Databricks</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=perplexity.ai&sz=128" height="40" alt="Perplexity"/><br/><sub><b>Perplexity</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=replicate.com&sz=128" height="40" alt="Replicate"/><br/><sub><b>Replicate</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=x.ai&sz=128" height="40" alt="xAI Grok"/><br/><sub><b>xAI (Grok)</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=baidu.com&sz=128" height="40" alt="Baidu ERNIE"/><br/><sub><b>Baidu ERNIE</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=deepseek.com&sz=128" height="40" alt="DeepSeek"/><br/><sub><b>DeepSeek</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=ollama.com&sz=128" height="40" alt="Ollama"/><br/><sub><b>Ollama</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=together.ai&sz=128" height="40" alt="Together AI"/><br/><sub><b>Together AI</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=openrouter.ai&sz=128" height="40" alt="OpenRouter"/><br/><sub><b>OpenRouter</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=fireworks.ai&sz=128" height="40" alt="Fireworks AI"/><br/><sub><b>Fireworks AI</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=cerebras.net&sz=128" height="40" alt="Cerebras"/><br/><sub><b>Cerebras</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=sambanova.ai&sz=128" height="40" alt="SambaNova"/><br/><sub><b>SambaNova</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=novita.ai&sz=128" height="40" alt="NovitaAI"/><br/><sub><b>NovitaAI</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=writer.com&sz=128" height="40" alt="Writer"/><br/><sub><b>Writer</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=ai21.com&sz=128" height="40" alt="AI21 Labs"/><br/><sub><b>AI21 Labs</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=aleph-alpha.com&sz=128" height="40" alt="Aleph Alpha"/><br/><sub><b>Aleph Alpha</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=minimax.io&sz=128" height="40" alt="Minimax"/><br/><sub><b>Minimax</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=moonshot.cn&sz=128" height="40" alt="Moonshot"/><br/><sub><b>Moonshot</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=zhipuai.cn&sz=128" height="40" alt="Zhipu"/><br/><sub><b>Zhipu</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=lmstudio.ai&sz=128" height="40" alt="LM Studio"/><br/><sub><b>LM Studio</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=ai.meta.com&sz=128" height="40" alt="llama.cpp"/><br/><sub><b>llama.cpp</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=docs.vllm.ai&sz=128" height="40" alt="vLLM"/><br/><sub><b>vLLM</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=gpt4all.io&sz=128" height="40" alt="GPT4All"/><br/><sub><b>GPT4All</b></sub></td>
  </tr>
</table>

---

### 🗄 Vector Stores — 22 backends

> All implement `VectorStore` with `add()`, `search()`, `search_mmr()`, `save()`, and `load()`. Built-in `InMemoryVectorStore` needs zero extra deps. Everything else is `pip install synapsekit[name]`.

<table>
  <tr>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=trychroma.com&sz=128" height="40" alt="ChromaDB"/><br/><sub><b>ChromaDB</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=ai.meta.com&sz=128" height="40" alt="FAISS"/><br/><sub><b>FAISS</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=qdrant.tech&sz=128" height="40" alt="Qdrant"/><br/><sub><b>Qdrant</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=pinecone.io&sz=128" height="40" alt="Pinecone"/><br/><sub><b>Pinecone</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=weaviate.io&sz=128" height="40" alt="Weaviate"/><br/><sub><b>Weaviate</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=milvus.io&sz=128" height="40" alt="Milvus"/><br/><sub><b>Milvus</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=lancedb.com&sz=128" height="40" alt="LanceDB"/><br/><sub><b>LanceDB</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=postgresql.org&sz=128" height="40" alt="PGVector"/><br/><sub><b>PGVector</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=sqlite.org&sz=128" height="40" alt="SQLiteVec"/><br/><sub><b>SQLiteVec</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=mongodb.com&sz=128" height="40" alt="MongoDB Atlas"/><br/><sub><b>MongoDB Atlas</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=redis.io&sz=128" height="40" alt="Redis"/><br/><sub><b>Redis</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=elastic.co&sz=128" height="40" alt="Elasticsearch"/><br/><sub><b>Elasticsearch</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=opensearch.org&sz=128" height="40" alt="OpenSearch"/><br/><sub><b>OpenSearch</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=supabase.com&sz=128" height="40" alt="Supabase"/><br/><sub><b>Supabase</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=cassandra.apache.org&sz=128" height="40" alt="Cassandra"/><br/><sub><b>Cassandra</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=duckdb.org&sz=128" height="40" alt="DuckDB"/><br/><sub><b>DuckDB</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=clickhouse.com&sz=128" height="40" alt="ClickHouse"/><br/><sub><b>ClickHouse</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=marqo.ai&sz=128" height="40" alt="Marqo"/><br/><sub><b>Marqo</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=typesense.org&sz=128" height="40" alt="Typesense"/><br/><sub><b>Typesense</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=docs.vespa.ai&sz=128" height="40" alt="Vespa"/><br/><sub><b>Vespa</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=zilliz.com&sz=128" height="40" alt="Zilliz"/><br/><sub><b>Zilliz</b></sub></td>
  </tr>
</table>

---

### 📂 Data Loaders — 64 sources

> All return `list[Document]` with `.text` and `.metadata`. Every loader has a sync `.load()` and async `.aload()`. Load from disk, cloud, databases, or APIs — same interface everywhere.

**File Formats**

<table>
  <tr>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=acrobat.adobe.com&sz=128" height="40" alt="PDF"/><br/><sub><b>PDF</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=word.office.com&sz=128" height="40" alt="Word"/><br/><sub><b>Word (DOCX)</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=excel.office.com&sz=128" height="40" alt="Excel"/><br/><sub><b>Excel (XLSX)</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=powerpoint.office.com&sz=128" height="40" alt="PowerPoint"/><br/><sub><b>PowerPoint</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=developer.mozilla.org&sz=128" height="40" alt="HTML"/><br/><sub><b>HTML / XML</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=markdownguide.org&sz=128" height="40" alt="Markdown"/><br/><sub><b>Markdown</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=latex-project.org&sz=128" height="40" alt="LaTeX"/><br/><sub><b>LaTeX</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=yaml.org&sz=128" height="40" alt="YAML"/><br/><sub><b>YAML / JSON</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=parquet.apache.org&sz=128" height="40" alt="Parquet"/><br/><sub><b>Parquet</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=openai.com&sz=128" height="40" alt="Audio"/><br/><sub><b>Audio (Whisper)</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=youtube.com&sz=128" height="40" alt="Video"/><br/><sub><b>Video</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=rss.com&sz=128" height="40" alt="RSS"/><br/><sub><b>RSS / Sitemap</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=git-scm.com&sz=128" height="40" alt="Git Repo"/><br/><sub><b>Git Repo</b></sub></td>
  </tr>
</table>

**Cloud Storage**

<table>
  <tr>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=aws.amazon.com&sz=128" height="40" alt="AWS S3"/><br/><sub><b>AWS S3</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=drive.google.com&sz=128" height="40" alt="Google Drive"/><br/><sub><b>Google Drive</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=azure.microsoft.com&sz=128" height="40" alt="Azure Blob"/><br/><sub><b>Azure Blob</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=onedrive.live.com&sz=128" height="40" alt="OneDrive"/><br/><sub><b>OneDrive</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=dropbox.com&sz=128" height="40" alt="Dropbox"/><br/><sub><b>Dropbox</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=cloud.google.com&sz=128" height="40" alt="GCS"/><br/><sub><b>Google Cloud</b></sub></td>
  </tr>
</table>

**Databases**

<table>
  <tr>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=postgresql.org&sz=128" height="40" alt="PostgreSQL"/><br/><sub><b>PostgreSQL</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=mysql.com&sz=128" height="40" alt="MySQL"/><br/><sub><b>MySQL</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=mongodb.com&sz=128" height="40" alt="MongoDB"/><br/><sub><b>MongoDB</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=aws.amazon.com&sz=128" height="40" alt="DynamoDB"/><br/><sub><b>DynamoDB</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=elastic.co&sz=128" height="40" alt="Elasticsearch"/><br/><sub><b>Elasticsearch</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=redis.io&sz=128" height="40" alt="Redis"/><br/><sub><b>Redis</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=cloud.google.com&sz=128" height="40" alt="BigQuery"/><br/><sub><b>BigQuery</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=snowflake.com&sz=128" height="40" alt="Snowflake"/><br/><sub><b>Snowflake</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=sqlite.org&sz=128" height="40" alt="SQLite"/><br/><sub><b>SQLite</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=supabase.com&sz=128" height="40" alt="Supabase"/><br/><sub><b>Supabase</b></sub></td>
  </tr>
</table>

**APIs & Productivity**

<table>
  <tr>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=github.com&sz=128" height="40" alt="GitHub"/><br/><sub><b>GitHub</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=atlassian.com&sz=128" height="40" alt="Jira"/><br/><sub><b>Jira</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=confluence.atlassian.com&sz=128" height="40" alt="Confluence"/><br/><sub><b>Confluence</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=notion.so&sz=128" height="40" alt="Notion"/><br/><sub><b>Notion</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=slack.com&sz=128" height="40" alt="Slack"/><br/><sub><b>Slack</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=discord.com&sz=128" height="40" alt="Discord"/><br/><sub><b>Discord</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=hubspot.com&sz=128" height="40" alt="HubSpot"/><br/><sub><b>HubSpot</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=salesforce.com&sz=128" height="40" alt="Salesforce"/><br/><sub><b>Salesforce</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=airtable.com&sz=128" height="40" alt="Airtable"/><br/><sub><b>Airtable</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=youtube.com&sz=128" height="40" alt="YouTube"/><br/><sub><b>YouTube</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=reddit.com&sz=128" height="40" alt="Reddit"/><br/><sub><b>Reddit</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=wikipedia.org&sz=128" height="40" alt="Wikipedia"/><br/><sub><b>Wikipedia</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=obsidian.md&sz=128" height="40" alt="Obsidian"/><br/><sub><b>Obsidian</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=sheets.google.com&sz=128" height="40" alt="Google Sheets"/><br/><sub><b>Google Sheets</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=firebase.google.com&sz=128" height="40" alt="Firebase"/><br/><sub><b>Firebase</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=twilio.com&sz=128" height="40" alt="Twilio"/><br/><sub><b>Twilio</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=arxiv.org&sz=128" height="40" alt="arXiv"/><br/><sub><b>arXiv</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=pubmed.ncbi.nlm.nih.gov&sz=128" height="40" alt="PubMed"/><br/><sub><b>PubMed</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=gmail.com&sz=128" height="40" alt="Email"/><br/><sub><b>Email (IMAP)</b></sub></td>
  </tr>
</table>

---

### 🔧 Agent Tools — 48+ built-in

> All implement `BaseTool` with a single async `run()`. Pass any list of tools to `ReActAgent` or `FunctionCallingAgent`. **Write your own in 5 lines.**

<table>
  <tr>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=duckduckgo.com&sz=128" height="40" alt="DuckDuckGo"/><br/><sub><b>DuckDuckGo</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=google.com&sz=128" height="40" alt="Google Search"/><br/><sub><b>Google Search</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=tavily.com&sz=128" height="40" alt="Tavily"/><br/><sub><b>Tavily</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=wolframalpha.com&sz=128" height="40" alt="Wolfram Alpha"/><br/><sub><b>Wolfram Alpha</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=wikipedia.org&sz=128" height="40" alt="Wikipedia"/><br/><sub><b>Wikipedia</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=youtube.com&sz=128" height="40" alt="YouTube"/><br/><sub><b>YouTube</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=arxiv.org&sz=128" height="40" alt="arXiv"/><br/><sub><b>arXiv</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=pubmed.ncbi.nlm.nih.gov&sz=128" height="40" alt="PubMed"/><br/><sub><b>PubMed</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=slack.com&sz=128" height="40" alt="Slack"/><br/><sub><b>Slack</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=discord.com&sz=128" height="40" alt="Discord"/><br/><sub><b>Discord</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=github.com&sz=128" height="40" alt="GitHub"/><br/><sub><b>GitHub API</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=atlassian.com&sz=128" height="40" alt="Jira"/><br/><sub><b>Jira</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=notion.so&sz=128" height="40" alt="Notion"/><br/><sub><b>Notion</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=linear.app&sz=128" height="40" alt="Linear"/><br/><sub><b>Linear</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=stripe.com&sz=128" height="40" alt="Stripe"/><br/><sub><b>Stripe</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=twilio.com&sz=128" height="40" alt="Twilio"/><br/><sub><b>Twilio</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=calendar.google.com&sz=128" height="40" alt="Google Calendar"/><br/><sub><b>Google Calendar</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=aws.amazon.com&sz=128" height="40" alt="AWS Lambda"/><br/><sub><b>AWS Lambda</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=playwright.dev&sz=128" height="40" alt="Browser"/><br/><sub><b>Browser (Playwright)</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=mysql.com&sz=128" height="40" alt="SQL"/><br/><sub><b>SQL Query</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=python.org&sz=128" height="40" alt="Python REPL"/><br/><sub><b>Python REPL</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=gnu.org&sz=128" height="40" alt="Shell"/><br/><sub><b>Shell</b></sub></td>
  </tr>
</table>

---

### 🧠 Memory & Cache Backends

<table>
  <tr>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=sqlite.org&sz=128" height="40" alt="SQLite"/><br/><sub><b>SQLite</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=redis.io&sz=128" height="40" alt="Redis"/><br/><sub><b>Redis</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=postgresql.org&sz=128" height="40" alt="PostgreSQL"/><br/><sub><b>PostgreSQL</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=aws.amazon.com&sz=128" height="40" alt="DynamoDB"/><br/><sub><b>DynamoDB</b></sub></td>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=memcached.org&sz=128" height="40" alt="Memcached"/><br/><sub><b>Memcached</b></sub></td>
  </tr>
</table>

### 📡 Observability

<table>
  <tr>
    <td align="center" width="90"><img src="https://www.google.com/s2/favicons?domain=opentelemetry.io&sz=128" height="40" alt="OpenTelemetry"/><br/><sub><b>OpenTelemetry</b></sub></td>
  </tr>
</table>

### Multi-Hop Knowledge Graph RAG

SynapseKit provides advanced retrieval modules, including vector search and multi-hop Knowledge Graph (KG) retrieval.

**When to use which?**
- **Vector Search (Semantic):** Best for broad conceptual queries, finding similar passages, or answering questions whose answers are contained within a single chunk of text.
- **Knowledge Graph (KG):** Best for specific, multi-hop reasoning questions where the relationship spans across multiple documents (e.g., finding out who owns the parent company of a subsidiary).
- **Hybrid (Vector + KG):** Combining both strategies guarantees that you capture deep semantic context while also exploring explicitly extracted entity relationships. Initialize the `RAG` facade with `graph_store=NetworkXStore()` or `Neo4jStore(...)` to enable this out-of-the-box.

<div align="center">

---
**Don't see your stack?**
Every integration is built the same way — most take under an hour.
[Browse `good first issue` →](https://github.com/SynapseKit/SynapseKit/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) · [Contributing guide →](CONTRIBUTING.md) · [Discord →](https://discord.gg/PSuAXHRywJ)

We credit every contributor in the README and send a personal thank-you on Discord.

</div>

---

## Install

**pip**
```bash
pip install synapsekit[openai]       # OpenAI
pip install synapsekit[anthropic]    # Anthropic
pip install synapsekit[ollama]       # Ollama (local)
pip install synapsekit[performance]  # orjson + uvloop + xxhash (faster)
pip install synapsekit[observe]      # Observability extras
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

Observability guide → [docs/observability.md](docs/observability.md)

---

## Documentation

Everything you need to get started and go deep is in the docs.

| | |
|---|---|
| 🚀 [Quickstart](https://synapsekit.github.io/synapsekit-docs/docs/getting-started/quickstart) | Up and running in 5 minutes |
| 🗂 [RAG](https://synapsekit.github.io/synapsekit-docs/docs/rag/pipeline) | Pipelines, loaders, retrieval, vector stores |
| 🤖 [Agents](https://synapsekit.github.io/synapsekit-docs/docs/agents/overview) | ReAct, function calling, tools, executor |
| 🔀 [Graph Workflows](https://synapsekit.github.io/synapsekit-docs/docs/graph/overview) | DAG pipelines, conditional routing, parallel execution |
| 🧠 [LLM Providers](https://synapsekit.github.io/synapsekit-docs/docs/llms/overview) | All 33 providers + ReasoningLLM with examples |
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
