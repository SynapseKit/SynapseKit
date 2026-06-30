from __future__ import annotations

import synapsekit.observe as observe
from synapsekit.observability.metrics import PrometheusMetrics


class TestPrometheusMetrics:
    def test_prometheus_metrics_records_from_span(self):
        metrics = PrometheusMetrics(enabled=False)
        observe.configure(metrics=metrics)

        span = observe.start_span(
            "llm.generate",
            {
                "llm.model": "gpt-4o-mini",
                "llm.provider": "openai",
                "llm.total_tokens": 12,
                "llm.cost_usd": 0.0012,
                "llm.latency_ms": 25.0,
            },
        )
        observe.end_span(span)

        # No crash, metrics disabled => no-ops
        assert True

    def test_agent_evolution_metrics_disabled_noop(self):
        metrics = PrometheusMetrics(enabled=False)

        metrics.record_agent_evolution_cycle(agent_id="agent", proposals=2)
        metrics.record_agent_evolution_patch(
            agent_id="agent",
            patch_type="prompt_rewrite",
            status="canary",
            eval_score=0.92,
            rollout_pct=5.0,
        )

        assert True
