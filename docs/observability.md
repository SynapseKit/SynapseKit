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

## Export to Grafana in 5 minutes

1) Start Prometheus + Grafana (see the Helm chart below or your own stack).
2) Enable metrics and expose a `/metrics` endpoint:

```python
from synapsekit.observability import PrometheusMetrics
import synapsekit.observe as observe

metrics = PrometheusMetrics(start_server=True, port=8000)
observe.configure(
    exporter="otlp",
    endpoint="http://localhost:4317",
    metrics=metrics,
)
```

3) In Grafana, add Prometheus as a data source.
4) Import `assets/grafana/synapsekit-observe-dashboard.json`.

## Prometheus metrics

SynapseKit emits the following metrics when enabled:

- `synapsekit_cost_usd_total` (counter)
- `synapsekit_tokens_total` (counter)
- `synapsekit_latency_seconds` (histogram)
- `synapsekit_budget_remaining_usd` (gauge)
- `synapsekit_arbitrage_savings_usd` (gauge)

## Cut LLM cost 40% in one line

Wrap existing provider clients without changing your application call site:

```python
from synapsekit import CostQualityRouter, PricingTable

router = CostQualityRouter(
    candidates=[economy_llm, premium_llm],
    pricing_table=PricingTable.from_cost_table(),
    explore_n=0,
)
answer = await router.generate("Summarise this document")
```

Use `ModelPricing` entries to override provider/model input, output, cached-input,
quality, latency-SLA, region, and carbon metadata. Request classes apply a
quality floor and maximum latency; the router chooses the cheapest eligible
provider and model. Pass `tenant_id` and `api_key_id` on each call to apply
`BudgetLedger` hard caps, soft alerts, and spend attribution. A
`FallbackChain` remains a valid candidate, so outage and rate-limit fallback
behavior composes with routing. Pass `PrometheusMetrics()` for cost, remaining
budget, and arbitrage-savings metrics, and pass `CarbonEstimator` only when
your organization has a preferred regional intensity table.

**Precondition:** once a `budget_ledger` is configured and a call's
`tenant_id`/`api_key_id` has a budget policy attached, every candidate model
routed to must have a registered `ModelPricing` entry. Requests to models
missing from the pricing table raise `BudgetExceededError("...pricing...")`
rather than silently treating them as free — register custom/self-hosted
model pricing with `pricing_table.register(ModelPricing(...))` before
enabling budgets for them.

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
