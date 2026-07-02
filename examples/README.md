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

### 6. `agent_memory.py` — Persistent Memory in Agents
Shows PR2-style memory integration:
- auto-recall injected into each turn
- episodic memory stored after each run
- `AgentExecutor` wiring with `PersistentAgentMemory`

```bash
python examples/agent_memory.py
```

### 7. `reasoning_models.py` — Reasoning LLMs *(v1.7.0)*
Use `ReasoningLLM` across OpenAI o1/o3, Claude thinking, Gemini thinking, DeepSeek R1, and Qwen QwQ.
Returns structured `ReasoningResponse` with answer, thinking trace, and token counts.

```bash
pip install synapsekit[openai,anthropic]
python examples/reasoning_models.py
```

### 8. `multimodal_rag.py` — Multimodal RAG
Load images, audio (Whisper transcription), and video into a single RAG pipeline.
Requires `ffmpeg` for video frame extraction.

```bash
pip install synapsekit[openai]
python examples/multimodal_rag.py
```

### 9. `fine_tune_flywheel.py` — Fine-Tune Data Flywheel *(v1.7.0)*
End-to-end workflow: capture eval results → filter dataset → export to OpenAI/Together AI format → submit fine-tune job.

```bash
pip install synapsekit[openai]
python examples/fine_tune_flywheel.py
```

### 10. `observability.py` — OpenTelemetry Observability
Trace RAG calls and LLM requests with OpenTelemetry. Works with Jaeger, Grafana Tempo, or any OTLP backend.

```bash
pip install synapsekit[observe]
python examples/observability.py
```

### 11. `agent_swarm_market.py` - Market-Based Agent Swarm
Route work across a five-agent fleet with sealed-bid market selection, deterministic bidding, reputation updates, and Mermaid trace output.

```bash
python examples/agent_swarm_market.py
```

## General Pattern

All examples follow this pattern:
- Use `os.environ` for API keys (never hardcode)
- Include docstrings explaining what the example does
- Work with minimal dependencies (`pip install synapsekit[openai]`)
- Print step-by-step progress for learning

## Contributing

Found an issue or want to add more examples? Open an issue or PR on GitHub!
