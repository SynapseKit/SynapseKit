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

## AG-UI Live stream

SynapseKit Live can expose its existing in-process event stream as AG-UI Server-Sent
Events. Install the protocol SDK as an optional extra:

```bash
pip install 'synapsekit[ag-ui]'
```

Start Live as usual, then connect a custom read-only AG-UI SSE consumer to
`GET /agui/events?token=<token>`. This is a passive event feed, not the standard
AG-UI HTTP-SSE agent binding: it does not accept `POST RunAgentInput`, and a
standard `HttpAgent` cannot use it to start a run. The per-process token is
included in the URL returned by `serve()` and used by the dashboard. The route
reuses Live's bounded
event history and subscriber queue, sends an initial connection comment and
periodic heartbeats, and unsubscribes when the client disconnects. Without the
extra, the endpoint returns `503` with the install instruction; importing
`synapsekit.live` still works without AG-UI installed.

The server binds to `127.0.0.1` by default. The AG-UI route is a read-only GET
stream of events already published in this process: it does not add a POST or
remote execution surface. The existing raw dashboard stream at `/events` is
unchanged. Do not expose the server on a non-loopback interface unless you have
a deliberate network-access policy; the token is a local access gate, not a
substitute for a trusted deployment boundary.

Supported event mapping:

| SynapseKit Live source | AG-UI output |
| --- | --- |
| `run.start` | `RUN_STARTED` |
| `tool.call` with `attributes.tool` | `TOOL_CALL_START`, `TOOL_CALL_END` |
| `span` named `llm.generate` with string `attributes["llm.output"]` | `TEXT_MESSAGE_START`, `TEXT_MESSAGE_CONTENT`, `TEXT_MESSAGE_END` |
| `span` or `tool.call` with `status="error"` | Generic `RUN_ERROR` without traceback/details |
| `graph.snapshot` with `attributes.nodes` and `attributes.edges` | First event: `STATE_SNAPSHOT`; later events: `STATE_DELTA` with an RFC 6902 `replace` operation at `/graph` |

Other event kinds are ignored. Tool arguments and results are not present in the
current Live tool event and therefore are not emitted. Text is forwarded only
when the supported `llm.output` source field contains an actual string; prompts,
tracebacks, and arbitrary span attributes are never copied into AG-UI events.
If LLM output tracing is enabled, that real output string will be visible to
connected clients, so apply the same privacy controls as for other Live output.

Live events do not currently provide thread, run, message, or tool-call IDs. The
adapter generates UUID correlation IDs required by AG-UI and treats them as
transport metadata, not source identities. No `RUN_FINISHED` event is inferred:
the current Live source has no matching run-end event, and a client disconnect
only removes its EventBus subscription.

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
