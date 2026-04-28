"""SynapseKit observability example.

Jaeger quickstart:

```yaml
services:
  jaeger:
    image: jaegertracing/all-in-one:1.57
    ports:
      - "16686:16686"
      - "4317:4317"
```

Then run:

    from synapsekit.observe import configure
    configure(exporter="jaeger", endpoint="http://localhost:4317")

Open Jaeger UI at http://localhost:16686.
"""

from __future__ import annotations

import asyncio

import synapsekit.observe as observe
from synapsekit import RAG


async def main() -> None:
    observe.configure(
        exporter="jaeger",
        endpoint="http://localhost:4317",
        service_name="synapsekit-observe-example",
        trace_llm_inputs=True,
        trace_llm_outputs=True,
        cost_tracking=True,
    )

    rag = RAG(model="gpt-4o-mini", api_key="sk-...", trace=True)
    rag.add("SynapseKit emits spans for LLM calls, retrieval, and graph execution.")
    answer = await rag.ask("What does SynapseKit trace?")

    print(answer)
    print(observe.get_exporter().export_dicts())


if __name__ == "__main__":
    asyncio.run(main())
