# Observability

`import synapsekit.observe` enables SynapseKit's tracing hooks for LLMs, RAG pipelines, agents, and graphs.

## Install

```bash
pip install synapsekit[observe,openai]
```

## Quickstart

```python
import synapsekit.observe as observe
from synapsekit import RAG

observe.configure(
    exporter="jaeger",
    endpoint="http://localhost:4317",
    service_name="my-rag-app",
    trace_llm_inputs=True,
    trace_llm_outputs=True,
    cost_tracking=True,
    sample_rate=1.0,
)

rag = RAG(model="gpt-4o-mini", api_key="sk-...")
answer = await rag.ask("What is the main topic?")
```

## Exporters

Supported exporter names:

- `console`
- `otlp`
- `jaeger`
- `langfuse`
- `honeycomb`

You can also pass a custom exporter object with `export()` and `clear()` methods.

## Privacy controls

```python
observe.configure(
    trace_llm_inputs=False,
    trace_llm_outputs=False,
    redact_keys=["api_key", "password"],
)
```

## Jaeger docker-compose snippet

```yaml
services:
  jaeger:
    image: jaegertracing/all-in-one:1.57
    ports:
      - "16686:16686"
      - "4317:4317"
```

## Grafana dashboard

Import `assets/grafana/synapsekit-observe-dashboard.json` into Grafana as a starting point for tracing dashboards.
