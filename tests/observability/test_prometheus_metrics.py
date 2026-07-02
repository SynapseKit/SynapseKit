from __future__ import annotations

import synapsekit.observe as observe
from synapsekit.observability.metrics import PrometheusMetrics


def test_prometheus_metrics_records_from_span():
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

    # disabled metrics: no-op, span must not raise and returns a non-None object
    assert span is not None
    assert isinstance(metrics, PrometheusMetrics)


def test_prometheus_metrics_records_swarm_noop_when_disabled():
    metrics = PrometheusMetrics(enabled=False)

    metrics.record_swarm_bid(
        agent_id="agent-a",
        task_category="research",
        auction_type="sealed_bid",
    )
    metrics.record_swarm_auction(
        auction_type="sealed_bid",
        task_category="research",
        bid_count=3,
        latency_ms=1.5,
    )
    metrics.record_swarm_win(
        agent_id="agent-a",
        task_category="research",
        auction_type="sealed_bid",
        reward=0.8,
        settlement_cost=12.0,
    )

    # All calls are no-ops when disabled; if they raised the test would fail
    assert metrics.enabled is False


def test_agent_evolution_metrics_disabled_noop():
    metrics = PrometheusMetrics(enabled=False)

    metrics.record_agent_evolution_cycle(agent_id="agent", proposals=2)
    metrics.record_agent_evolution_patch(
        agent_id="agent",
        patch_type="prompt_rewrite",
        status="canary",
        eval_score=0.92,
        rollout_pct=5.0,
    )

    # All calls are no-ops when disabled; if they raised the test would fail
    assert metrics.enabled is False
