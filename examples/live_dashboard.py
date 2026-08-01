"""Watch a SynapseKit run live in your browser — zero extra dependencies.

    python examples/live_dashboard.py

Opens http://127.0.0.1:7900 and streams a simulated support-agent run into the
SynapseKit Live dashboard (stdlib http.server + Server-Sent Events, no FastAPI /
uvicorn / websockets). In real code you don't script events — just set
``SYNAPSEKIT_LIVE=1`` (or call ``synapsekit.live.enable()``) and every span your
agents/RAG/graph produce shows up here automatically.
"""

from __future__ import annotations

import time

from synapsekit.live import bus, serve


def _span(name: str, status: str, ms: float, **attributes: object) -> None:
    bus.publish(
        {"kind": "span", "name": name, "status": status, "duration_ms": ms, "attributes": attributes}
    )


def main() -> None:
    serve(open_browser=True)  # prints the URL and opens a tab
    print("Streaming a demo run… (Ctrl+C to stop)")

    run = [
        ("agent.run", 40, {"channel": "support-desk"}),
        ("retriever.search", 42, {"hits": 5, "store": "pgvector", "top_score": 0.91}),
        ("llm.call", 980, {"model": "claude-opus-4-8", "provider": "anthropic", "tokens_in": 312}),
        ("tool.call", 28, {"tool": "lookup_order", "ok": True}),
        ("llm.call", 640, {"model": "claude-opus-4-8", "tokens_out": 148, "cost_usd": 0.0043}),
        ("tool.call", 15, {"tool": "send_email", "sent": 202}),
        ("run.complete", 5, {"cost_usd": 0.0043, "total_ms": 1900}),
    ]

    try:
        while True:
            for name, ms, attrs in run:
                _span(name, "ok", ms, **attrs)
                time.sleep(0.9)
            time.sleep(2.5)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
