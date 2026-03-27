# Examples

This directory contains runnable examples demonstrating key SynapseKit features.

## Prerequisites

Install SynapseKit with OpenAI support:
```bash
pip install synapsekit[openai]
```

Set your API key:
```bash
export OPENAI_API_KEY=sk-...
```

## Examples

### 1. `rag_quickstart.py` — RAG Basics
The simplest way to get started: load text, add documents, and query with streaming.

```bash
python examples/rag_quickstart.py
```

### 2. `agent_tools.py` — ReAct Agent with Tools
Create a ReAct agent with built-in and custom tools. Shows reasoning and tool execution.

```bash
python examples/agent_tools.py
```

### 3. `graph_workflow.py` — State Graph with Conditional Routing
Build workflows with state management, conditional edges, and visualization.

```bash
python examples/graph_workflow.py
```

### 4. `multi_provider.py` — Multi-Provider Comparison
Run the same prompt across OpenAI, Anthropic, and Ollama to compare responses.

Requires additional setup:
```bash
pip install synapsekit[openai,anthropic]
export ANTHROPIC_API_KEY=sk-ant-...
```

```bash
python examples/multi_provider.py
```

### 5. `caching_retries.py` — Advanced LLM Configuration
Configure response caching, automatic retries, and cost tracking with budget limits.

```bash
python examples/caching_retries.py
```

## General Pattern

All examples follow this pattern:
- Use `os.environ` for API keys (never hardcode)
- Include docstrings explaining what the example does
- Work with minimal dependencies (`pip install synapsekit[openai]`)
- Print step-by-step progress for learning

## Contributing

Found an issue or want to add more examples? Open an issue or PR on GitHub!
