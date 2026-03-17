# Contributing to SynapseKit

Thank you for your interest in contributing. SynapseKit is an open source project and contributions of all kinds are welcome — bug reports, documentation improvements, new features, and code reviews.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Ways to Contribute](#ways-to-contribute)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Making Changes](#making-changes)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Coding Standards](#coding-standards)
- [Writing Tests](#writing-tests)
- [Commit Message Format](#commit-message-format)

---

## Code of Conduct

This project follows our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold it.

---

## Ways to Contribute

**Not sure where to start?** Look for issues tagged [`good first issue`](https://github.com/SynapseKit/SynapseKit/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) or [`help wanted`](https://github.com/SynapseKit/SynapseKit/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22).

| Type | How |
|---|---|
| 🐛 **Bug report** | [Open an issue](https://github.com/SynapseKit/SynapseKit/issues/new?template=bug_report.yml) |
| 💡 **Feature request** | [Open an issue](https://github.com/SynapseKit/SynapseKit/issues/new?template=feature_request.yml) |
| 📖 **Documentation** | Edit docs in [synapsekit-docs](https://github.com/SynapseKit/synapsekit-docs) or fix docstrings here |
| 🔌 **New LLM provider** | See [Adding a Provider](#adding-a-provider) |
| 🗄️ **New vector store** | See [Adding a Vector Store](#adding-a-vector-store) |
| 🛠️ **New tool** | See [Adding a Tool](#adding-a-tool) |

---

## Development Setup

SynapseKit uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# 1. Fork and clone
git clone https://github.com/<your-username>/SynapseKit
cd SynapseKit

# 2. Install dependencies
uv sync --group dev

# 3. Run the test suite
uv run pytest tests/ -q

# 4. Create a branch
git checkout -b feat/my-feature
```

All 267 tests should pass before you start. If any are failing, open an issue.

---

## Project Structure

```
synapsekit/
├── _compat.py          # run_sync() — works inside/outside event loops
├── agents/             # ReActAgent, FunctionCallingAgent, AgentExecutor, tools
├── embeddings/         # SynapsekitEmbeddings (sentence-transformers backend)
├── graph/              # StateGraph, CompiledGraph, graph workflows
├── llm/                # BaseLLM + provider implementations (openai, anthropic, …)
├── loaders/            # Document loaders (pdf, html, csv, json, web, directory)
├── memory/             # ConversationMemory
├── observability/      # TokenTracer
├── parsers/            # JSONParser, ListParser, PydanticParser
├── prompts/            # PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate
├── rag/                # RAGPipeline, TextSplitter
├── retrieval/          # VectorStore ABC, InMemoryVectorStore, Chroma, FAISS, Qdrant, Pinecone
tests/
├── test_*.py           # One file per module
```

---

## Making Changes

1. **Start with an issue.** For anything beyond a typo fix, open an issue first to discuss the approach. This avoids wasted effort.

2. **Keep changes focused.** One logical change per pull request. Don't refactor unrelated code alongside a bug fix.

3. **Write tests.** All new code must be covered by tests. See [Writing Tests](#writing-tests).

4. **Update docs.** If you add a public API, update the [docs site](https://github.com/SynapseKit/synapsekit-docs).

---

## Pull Request Guidelines

- Target the `main` branch
- Fill in the pull request template completely
- Link the related issue with `Closes #<issue>`
- All tests must pass: `uv run pytest tests/ -q`
- Keep the PR description clear — explain *why*, not just *what*
- Request review from a maintainer once ready

---

## Coding Standards

- **Async-first.** Public APIs must be `async`. Provide sync wrappers via `run_sync()` where appropriate.
- **Type hints.** All public functions and methods must be fully typed.
- **No new hard dependencies.** Core functionality must work with `numpy` and `rank-bm25` only. New providers and backends go behind optional extras.
- **No magic.** No monkey-patching, hidden callbacks, or implicit global state.
- **Python 3.10+.** Use modern syntax freely.

---

## Writing Tests

Tests live in `tests/`. One file per module, named `test_<module>.py`.

```bash
# Run all tests
uv run pytest tests/ -q

# Run a specific file
uv run pytest tests/test_graph_run.py -v

# Run a specific test
uv run pytest tests/test_graph_run.py::test_linear_two_nodes -v
```

- Use `pytest-asyncio` for async tests — mark with `async def test_...` (auto mode is on)
- Mock external APIs — no test should require an API key or network access
- Test both the happy path and error cases
- Keep tests fast — the full suite should run in under 5 seconds

---

## Commit Message Format

We use a simplified [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>: <short description>

[optional body]
```

| Type | When to use |
|---|---|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `test` | Adding or updating tests |
| `refactor` | Code change that is not a fix or feature |
| `chore` | Build tooling, dependencies, CI |

**Examples:**
```
feat(graph): add ConditionalEdge async condition support
fix(llm): handle empty response from Anthropic stream
docs: add Graph Workflows section to README
test(agents): add FunctionCallingAgent multi-tool tests
```

---

## Adding a Provider

1. Create `synapsekit/llm/<provider>.py` — subclass `BaseLLM`, implement `stream()`
2. Add to `_make_llm()` in `synapsekit/__init__.py`
3. Add optional dependency to `pyproject.toml`
4. Add tests in `tests/test_llm_providers.py`
5. Add a doc page in `synapsekit-docs/docs/llms/<provider>.md`

---

## Adding a Vector Store

1. Create `synapsekit/retrieval/<backend>.py` — subclass `VectorStore`, implement `add()`, `search()`, `save()`, `load()`
2. Add optional dependency to `pyproject.toml`
3. Add tests in `tests/test_vectorstore_backends.py`

---

## Adding a Tool

1. Create a class in `synapsekit/agents/tools.py` — subclass `BaseTool`, implement `run()`
2. Export from `synapsekit/agents/__init__.py` and `synapsekit/__init__.py`
3. Add tests in `tests/test_tools.py`

---

## Questions?

Open a [discussion](https://github.com/SynapseKit/SynapseKit/discussions) or [issue](https://github.com/SynapseKit/SynapseKit/issues). We're happy to help.
