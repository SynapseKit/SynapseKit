"""SynapseKit observability dashboard — FastAPI server."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI

_DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>SynapseKit Observability Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            padding: 24px;
        }
        h1 { color: #58a6ff; font-size: 1.6rem; margin-bottom: 8px; }
        .subtitle { color: #8b949e; font-size: 0.9rem; margin-bottom: 24px; }
        .badge {
            display: inline-block;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 2px 10px;
            font-size: 0.75rem;
            color: #58a6ff;
            margin-left: 8px;
        }
        .metrics-row {
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
            margin-bottom: 28px;
        }
        .metric-card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 16px 24px;
            min-width: 150px;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #58a6ff;
        }
        .metric-label {
            font-size: 0.8rem;
            color: #8b949e;
            margin-top: 4px;
        }
        section { margin-bottom: 32px; }
        section h2 {
            font-size: 1.1rem;
            color: #e6edf3;
            margin-bottom: 12px;
            border-bottom: 1px solid #30363d;
            padding-bottom: 6px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }
        th {
            text-align: left;
            padding: 8px 12px;
            background: #161b22;
            color: #8b949e;
            font-weight: 600;
            border-bottom: 1px solid #30363d;
        }
        td {
            padding: 8px 12px;
            border-bottom: 1px solid #21262d;
            color: #c9d1d9;
        }
        tr:hover td { background: #1c2128; }
        .tag {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
        }
        .tag-ok { background: #0d4429; color: #3fb950; }
        .tag-warn { background: #3d2400; color: #d29922; }
        .tag-bad { background: #2d0101; color: #f85149; }
        .empty { color: #484f58; font-style: italic; padding: 16px 12px; }
        #refresh-badge {
            float: right;
            font-size: 0.78rem;
            color: #484f58;
        }
        .step-bar {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 6px;
        }
        .step-label { font-size: 0.82rem; min-width: 140px; color: #8b949e; }
        .step-fill {
            height: 14px;
            background: #1f6feb;
            border-radius: 3px;
            min-width: 4px;
        }
        .step-val { font-size: 0.78rem; color: #8b949e; }
    </style>
</head>
<body>
    <h1>SynapseKit <span class="badge">Observability</span></h1>
    <p class="subtitle">
        Live dashboard — LLM traces, RAG metrics, agent timelines.
        <span id="refresh-badge">Auto-refreshing every 5s</span>
    </p>

    <div class="metrics-row" id="metric-cards">
        <!-- filled by JS -->
    </div>

    <section>
        <h2>Recent LLM Traces (last 100)</h2>
        <table id="traces-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Timestamp</th>
                    <th>Model</th>
                    <th>Prompt Tokens</th>
                    <th>Completion Tokens</th>
                    <th>Latency (ms)</th>
                    <th>Est. Cost (USD)</th>
                </tr>
            </thead>
            <tbody id="traces-body">
                <tr><td colspan="7" class="empty">Loading...</td></tr>
            </tbody>
        </table>
    </section>

    <section>
        <h2>RAG Pipeline Metrics</h2>
        <div id="rag-metrics">Loading...</div>
    </section>

    <section>
        <h2>Agent Execution Timeline</h2>
        <div id="agent-timeline">Loading...</div>
    </section>

    <script>
        function fmtTs(ts) {
            if (!ts) return '—';
            return new Date(ts * 1000).toLocaleTimeString();
        }
        function fmtScore(v) {
            if (v === null || v === undefined) return '—';
            const n = parseFloat(v);
            const cls = n >= 0.8 ? 'tag-ok' : n >= 0.5 ? 'tag-warn' : 'tag-bad';
            return '<span class="tag ' + cls + '">' + n.toFixed(3) + '</span>';
        }

        async function refreshAll() {
            try {
                const [tracesResp, metricsResp] = await Promise.all([
                    fetch('/api/traces'),
                    fetch('/api/metrics'),
                ]);
                const traces = await tracesResp.json();
                const metrics = await metricsResp.json();
                renderMetricCards(metrics);
                renderTraces(traces);
                renderRagMetrics(metrics);
                renderTimeline(traces);
            } catch (e) {
                console.error('Dashboard refresh error:', e);
            }
        }

        function renderMetricCards(m) {
            const cards = [
                { value: m.total_calls ?? 0, label: 'Total LLM Calls' },
                { value: (m.total_tokens ?? 0).toLocaleString(), label: 'Total Tokens' },
                { value: '$' + (m.total_cost_usd ?? 0).toFixed(4), label: 'Est. Cost (USD)' },
                { value: (m.avg_latency_ms ?? 0).toFixed(0) + 'ms', label: 'Avg Latency' },
                { value: m.avg_faithfulness != null ? m.avg_faithfulness.toFixed(3) : '—', label: 'Avg Faithfulness' },
                { value: m.avg_relevancy != null ? m.avg_relevancy.toFixed(3) : '—', label: 'Avg Relevancy' },
            ];
            document.getElementById('metric-cards').innerHTML = cards.map(c =>
                '<div class="metric-card"><div class="metric-value">' + c.value + '</div><div class="metric-label">' + c.label + '</div></div>'
            ).join('');
        }

        function renderTraces(traces) {
            const tbody = document.getElementById('traces-body');
            if (!traces.length) {
                tbody.innerHTML = '<tr><td colspan="7" class="empty">No traces recorded yet.</td></tr>';
                return;
            }
            tbody.innerHTML = traces.map((t, i) =>
                '<tr>' +
                '<td>' + (i + 1) + '</td>' +
                '<td>' + fmtTs(t.timestamp) + '</td>' +
                '<td>' + (t.model || '—') + '</td>' +
                '<td>' + (t.input_tokens ?? '—') + '</td>' +
                '<td>' + (t.output_tokens ?? '—') + '</td>' +
                '<td>' + (t.latency_ms != null ? t.latency_ms.toFixed(1) : '—') + '</td>' +
                '<td>' + (t.cost_usd != null ? '$' + t.cost_usd.toFixed(6) : '—') + '</td>' +
                '</tr>'
            ).join('');
        }

        function renderRagMetrics(m) {
            const el = document.getElementById('rag-metrics');
            const rows = [
                ['Avg Faithfulness', fmtScore(m.avg_faithfulness)],
                ['Avg Relevancy', fmtScore(m.avg_relevancy)],
                ['Quality Trend', m.quality_trend || '—'],
                ['Total Quality Records', m.total_quality_records ?? 0],
            ];
            el.innerHTML = '<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>' +
                rows.map(r => '<tr><td>' + r[0] + '</td><td>' + r[1] + '</td></tr>').join('') +
                '</tbody></table>';
        }

        function renderTimeline(traces) {
            const el = document.getElementById('agent-timeline');
            if (!traces.length) {
                el.innerHTML = '<p class="empty">No agent steps recorded yet.</p>';
                return;
            }
            const maxLatency = Math.max(...traces.map(t => t.latency_ms || 0)) || 1;
            el.innerHTML = traces.slice(0, 20).map((t, i) => {
                const pct = Math.max(4, ((t.latency_ms || 0) / maxLatency) * 300);
                return '<div class="step-bar">' +
                    '<span class="step-label">' + (t.model || 'call-' + (i + 1)) + '</span>' +
                    '<div class="step-fill" style="width:' + pct + 'px"></div>' +
                    '<span class="step-val">' + (t.latency_ms != null ? t.latency_ms.toFixed(0) + 'ms' : '—') + '</span>' +
                    '</div>';
            }).join('');
        }

        refreshAll();
        setInterval(refreshAll, 5000);
    </script>
</body>
</html>
"""


def _build_trace_list(tracer: Any) -> list[dict[str, Any]]:
    """Convert TokenTracer records to a JSON-serialisable list."""
    from synapsekit.observability.tracer import COST_TABLE

    costs = COST_TABLE.get(tracer.model, {})
    records = tracer._records[-100:]  # last 100
    result = []

    for i, rec in enumerate(records):
        cost = rec.input_tokens * costs.get("input", 0.0) + rec.output_tokens * costs.get(
            "output", 0.0
        )
        # quality record aligned by index if available
        q_rec = None
        if i < len(tracer._quality_records):
            q_rec = tracer._quality_records[i]
        result.append(
            {
                "index": i + 1,
                "timestamp": q_rec.timestamp if q_rec else None,
                "model": tracer.model,
                "input_tokens": rec.input_tokens,
                "output_tokens": rec.output_tokens,
                "latency_ms": rec.latency_ms,
                "cost_usd": cost,
            }
        )
    return result


def create_app(tracer: Any | None = None) -> FastAPI:
    """Create the observability dashboard FastAPI app.

    Args:
        tracer: An optional ``TokenTracer`` instance. If None, a default one is created
                and accessible via ``app.state.tracer``.
    """
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    from synapsekit.observability.tracer import COST_TABLE, TokenTracer

    app = FastAPI(title="SynapseKit Observability Dashboard")

    if tracer is None:
        tracer = TokenTracer(model="gpt-4o-mini", enabled=True)

    app.state.tracer = tracer

    @app.get("/")
    def dashboard() -> HTMLResponse:
        return HTMLResponse(content=_DASHBOARD_HTML)

    @app.get("/api/health")
    def health() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.get("/api/traces")
    def get_traces() -> JSONResponse:
        t = app.state.tracer
        return JSONResponse(_build_trace_list(t))

    @app.get("/api/metrics")
    def get_metrics() -> JSONResponse:
        t = app.state.tracer
        summary = t.summary()
        costs = COST_TABLE.get(t.model, {})
        records = t._records

        total_cost = sum(
            r.input_tokens * costs.get("input", 0.0) + r.output_tokens * costs.get("output", 0.0)
            for r in records
        )
        avg_latency = (
            sum(r.latency_ms for r in records) / len(records) if records else 0.0
        )

        return JSONResponse(
            {
                "total_calls": summary["calls"],
                "total_tokens": summary["total_tokens"],
                "total_input_tokens": summary["total_input_tokens"],
                "total_output_tokens": summary["total_output_tokens"],
                "total_cost_usd": round(total_cost, 6),
                "avg_latency_ms": round(avg_latency, 2),
                "avg_faithfulness": summary["avg_faithfulness"],
                "avg_relevancy": summary["avg_relevancy"],
                "quality_trend": summary["quality_trend"],
                "total_quality_records": len(t._quality_records),
            }
        )

    return app
