# Comparison Matrix — SynapseKit vs LangChain vs LlamaIndex vs Haystack

**Goal:** an honest, source-backed feature × framework matrix.

**Legend:** ✅ built-in/maintained · 🧩 optional plugin/integration · ⚠️ partial/experimental · ❌ not supported · 🚧 unknown/needs verification

**Last updated:** 2026-05-28

---

## Feature Grid (core capabilities)

| Feature | SynapseKit | LangChain | LlamaIndex | Haystack | Sources |
|---|---|---|---|---|---|
| RAG pipelines | ✅ | ✅ | ✅ | ✅ | SK1, LC2, LI2, HS1 |
| Agents | ✅ | ✅ | ✅ | ✅ | SK1, LC1, LI3, HS1 |
| Workflow / graph orchestration | ✅ (built-in) | 🧩 (LangGraph) | 🧩 (Workflows) | ✅ (pipelines + agent workflows) | SK1, LC1, LI4, HS1 |
| Streaming support | ✅ (default) | ✅ | ✅ | ✅ (streaming_callback) | SK1, LC3, LI2, HS2 |
| Async support | ✅ (async-first) | ⚠️ (async APIs; not default) | ✅ (async-first workflows) | 🚧 | SK1, LC3, LI4 |
| Cost tracking | ✅ | 🚧 | 🚧 | 🚧 | SK2 |
| Evaluation tooling | ✅ (EvalCI) | 🧩 (LangSmith evals) | ✅ | ✅ | SK3, LC1, LI5, HS1 |
| Observability / tracing | ✅ (OTel + metrics) | 🧩 (LangSmith) | 🧩 (Workflows + OTel) | 🧩 (Enterprise platform) | SK4, LC1, LI6, HS1 |
| Integrations ecosystem | 46 LLMs · 11 vector stores · 53 loaders · 47+ tools | Many integrations (see overview) | 300+ integration packages | Vendor-agnostic providers & components | SK1, LC1, LI1, HS1 |

---

## DX Comparison (developer experience)

| DX signal | SynapseKit | LangChain | LlamaIndex | Haystack | Sources |
|---|---|---|---|---|---|
| Minimal agent example LOC | **11** | **13** | **16** | **16** | SK1, LC1, LI3, HS2 |
| Async ergonomics | Async-first by default | Async support; not default | Async-first in workflows | 🚧 | SK1, LC3, LI4 |
| Typing / mypy signals | 🚧 | 🚧 | 🚧 | ✅ (mypy badge) | HS1 |

**LOC method:** counts non-empty, non-comment lines from the cited “minimal agent” snippet.

---

## License + Governance

| Item | SynapseKit | LangChain | LlamaIndex | Haystack | Sources |
|---|---|---|---|---|---|
| License | Apache-2.0 | MIT | MIT | Apache-2.0 | SK5, LC5, LI7, HS3 |
| Contribution guide | ✅ | ✅ | ✅ | ✅ | SK6, LC4, LI8, HS4 |
| Governance | OSS with public issues + contributions | OSS with public issues + contributions | OSS with public issues + contributions | OSS with public issues + contributions | SK6, LC4, LI8, HS4 |

---

## Maintenance

| Framework | Maintained | Last push (UTC) | Source |
|---|---|---|---|
| SynapseKit | ✅ | 2026-05-27 | GH1 |
| LangChain | ✅ | 2026-05-27 | GH2 |
| LlamaIndex | ✅ | 2026-05-26 | GH3 |
| Haystack | ✅ | 2026-05-28 | GH4 |

---

## Update cadence (automation)

- Quarterly refresh via `.github/workflows/comparison-matrix-update.yml`
- Script: `scripts/update_comparison_matrix.py`

---

## Sources

### SynapseKit
- SK1 — `README.md` (capabilities, counts, agent example)
- SK2 — `README.md` (“SynapseKit vs LangChain vs LlamaIndex” table; cost tracking)
- SK3 — `README.md` (EvalCI section)
- SK4 — `docs/observability.md`
- SK5 — `LICENSE` (Apache-2.0)
- SK6 — `CONTRIBUTING.md`

### LangChain
- LC1 — https://raw.githubusercontent.com/langchain-ai/langchain/master/README.md
- LC2 — https://docs.langchain.com/oss/python/langchain/rag.md
- LC3 — https://docs.langchain.com/oss/python/langchain/streaming.md
- LC4 — https://docs.langchain.com/oss/python/contributing/overview
- LC5 — https://raw.githubusercontent.com/langchain-ai/langchain/master/LICENSE

### LlamaIndex
- LI1 — https://raw.githubusercontent.com/run-llama/llama_index/main/README.md
- LI2 — https://developers.llamaindex.ai/api/read?path=/python/framework/module_guides/deploying/query_engine/
- LI3 — https://developers.llamaindex.ai/api/read?path=/python/framework/module_guides/deploying/agents/
- LI4 — https://developers.llamaindex.ai/api/read?path=/python/llamaagents/workflows/
- LI5 — https://developers.llamaindex.ai/api/read?path=/python/framework/module_guides/evaluating/
- LI6 — https://developers.llamaindex.ai/api/read?path=/python/llamaagents/workflows/observability/
- LI7 — https://raw.githubusercontent.com/run-llama/llama_index/main/LICENSE
- LI8 — https://raw.githubusercontent.com/run-llama/llama_index/main/CONTRIBUTING.md

### Haystack
- HS1 — https://raw.githubusercontent.com/deepset-ai/haystack/main/README.md
- HS2 — https://haystack.deepset.ai/overview/quick-start
- HS3 — https://raw.githubusercontent.com/deepset-ai/haystack/main/LICENSE
- HS4 — https://raw.githubusercontent.com/deepset-ai/haystack/main/CONTRIBUTING.md

### GitHub activity
- GH1 — https://api.github.com/repos/synapsekit/synapsekit
- GH2 — https://api.github.com/repos/langchain-ai/langchain
- GH3 — https://api.github.com/repos/run-llama/llama_index
- GH4 — https://api.github.com/repos/deepset-ai/haystack
