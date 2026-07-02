"""Prometheus metrics exporter for SynapseKit observability."""

from __future__ import annotations

from contextlib import suppress
from typing import Any

try:  # pragma: no cover - optional dependency
    from prometheus_client import (
        CollectorRegistry as PromCollectorRegistry,
    )
    from prometheus_client import (
        Counter as PromCounter,
    )
    from prometheus_client import (
        Histogram as PromHistogram,
    )
    from prometheus_client import (
        start_http_server as prom_start_http_server,
    )

    _PROMETHEUS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    PromCollectorRegistry = None  # type: ignore[assignment,misc]
    PromCounter = None  # type: ignore[assignment,misc]
    PromHistogram = None  # type: ignore[assignment,misc]
    prom_start_http_server = None  # type: ignore[assignment,misc]
    _PROMETHEUS_AVAILABLE = False

_EVOLUTION_STATUSES = {
    "proposed": 0.0,
    "blocked": 1.0,
    "approved": 2.0,
    "canary": 3.0,
    "promoted": 4.0,
    "rolled_back": 5.0,
    "rejected": 6.0,
}


class PrometheusMetrics:
    """Prometheus metrics for LLM cost, tokens, and latency.

    Metrics:
      - synapsekit_cost_usd_total
      - synapsekit_tokens_total
      - synapsekit_latency_seconds (histogram)
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        namespace: str = "synapsekit",
        registry: Any | None = None,
        start_server: bool = False,
        host: str = "0.0.0.0",
        port: int = 8000,
    ) -> None:
        self.enabled = bool(enabled) and _PROMETHEUS_AVAILABLE
        self._namespace = namespace
        self._registry: Any | None = registry
        if self._registry is None and self.enabled and PromCollectorRegistry is not None:
            self._registry = PromCollectorRegistry()
        self._server_started = False
        self._cost_counter: Any | None = None
        self._token_counter: Any | None = None
        self._latency_hist: Any | None = None
        self._swarm_bid_counter: Any | None = None
        self._swarm_win_counter: Any | None = None
        self._swarm_reward_hist: Any | None = None
        self._swarm_settlement_hist: Any | None = None
        self._swarm_auction_latency_hist: Any | None = None
        self._swarm_auction_bid_hist: Any | None = None
        self._agent_evolution_cycles: Any | None = None
        self._agent_evolution_patches: Any | None = None
        self._agent_evolution_eval_score: Any | None = None
        self._agent_evolution_rollout_pct: Any | None = None
        self._agent_evolution_status: Any | None = None

        if self.enabled:
            self._cost_counter = PromCounter(
                "cost_usd_total",
                "Total LLM cost in USD.",
                ["model", "provider"],
                namespace=self._namespace,
                registry=self._registry,
            )
            self._token_counter = PromCounter(
                "tokens_total",
                "Total LLM tokens.",
                ["model", "provider"],
                namespace=self._namespace,
                registry=self._registry,
            )
            self._latency_hist = PromHistogram(
                "latency_seconds",
                "LLM latency in seconds.",
                ["model", "provider"],
                namespace=self._namespace,
                registry=self._registry,
                buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
            )
            self._swarm_bid_counter = PromCounter(
                "swarm_bids_total",
                "Total agent swarm bids.",
                ["agent_id", "task_category", "auction_type"],
                namespace=self._namespace,
                registry=self._registry,
            )
            self._swarm_win_counter = PromCounter(
                "swarm_wins_total",
                "Total agent swarm auction wins.",
                ["agent_id", "task_category", "auction_type"],
                namespace=self._namespace,
                registry=self._registry,
            )
            self._swarm_reward_hist = PromHistogram(
                "swarm_reward",
                "Agent swarm reward per winning execution.",
                ["agent_id", "task_category", "auction_type"],
                namespace=self._namespace,
                registry=self._registry,
                buckets=(-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 2.0),
            )
            self._swarm_settlement_hist = PromHistogram(
                "swarm_settlement_cost",
                "Agent swarm settlement cost per winning execution.",
                ["agent_id", "task_category", "auction_type"],
                namespace=self._namespace,
                registry=self._registry,
                buckets=(0, 1, 10, 100, 1_000, 10_000, 100_000),
            )
            self._swarm_auction_latency_hist = PromHistogram(
                "swarm_auction_latency_seconds",
                "Agent swarm auction latency in seconds.",
                ["task_category", "auction_type"],
                namespace=self._namespace,
                registry=self._registry,
                buckets=(0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1),
            )
            self._swarm_auction_bid_hist = PromHistogram(
                "swarm_auction_bids",
                "Number of bids evaluated per swarm auction.",
                ["task_category", "auction_type"],
                namespace=self._namespace,
                registry=self._registry,
                buckets=(1, 2, 3, 5, 8, 13, 21, 34),
            )
            self._agent_evolution_cycles = PromCounter(
                "agent_evolution_cycles_total",
                "Total self-improvement cycles executed.",
                ["agent_id"],
                namespace=self._namespace,
                registry=self._registry,
            )
            self._agent_evolution_patches = PromCounter(
                "agent_evolution_patches_total",
                "Total self-improvement patches by type and status.",
                ["agent_id", "patch_type", "status"],
                namespace=self._namespace,
                registry=self._registry,
            )
            self._agent_evolution_eval_score = PromHistogram(
                "agent_evolution_eval_score",
                "Eval score for agent evolution patches.",
                ["agent_id", "patch_type", "status"],
                namespace=self._namespace,
                registry=self._registry,
                buckets=(0.0, 0.1, 0.25, 0.5, 0.7, 0.85, 0.9, 0.95, 1.0),
            )
            self._agent_evolution_rollout_pct = PromHistogram(
                "agent_evolution_rollout_pct",
                "Rollout percentage assigned to agent evolution patches.",
                ["agent_id", "patch_type", "status"],
                namespace=self._namespace,
                registry=self._registry,
                buckets=(0, 1, 5, 10, 25, 50, 75, 100),
            )
            self._agent_evolution_status = PromHistogram(
                "agent_evolution_status",
                "Numeric status code for agent evolution patches.",
                ["agent_id", "patch_type"],
                namespace=self._namespace,
                registry=self._registry,
                buckets=(0, 1, 2, 3, 4, 5, 6),
            )

        if start_server and self.enabled:
            self.start_http_server(host=host, port=port)

    def start_http_server(self, *, host: str = "0.0.0.0", port: int = 8000) -> None:
        if not self.enabled or self._server_started:
            return
        if prom_start_http_server is None or self._registry is None:
            return
        prom_start_http_server(port, addr=host, registry=self._registry)
        self._server_started = True

    def record_llm(
        self,
        *,
        model: str,
        provider: str,
        cost_usd: float | None,
        total_tokens: int | None,
        latency_ms: float | None,
    ) -> None:
        if not self.enabled:
            return
        if model is None:
            model = "unknown"
        if provider is None:
            provider = "unknown"
        labels = {"model": str(model), "provider": str(provider)}

        if cost_usd is not None and self._cost_counter is not None:
            with suppress(Exception):
                self._cost_counter.labels(**labels).inc(float(cost_usd))

        if total_tokens is not None and self._token_counter is not None:
            with suppress(Exception):
                self._token_counter.labels(**labels).inc(int(total_tokens))

        if latency_ms is not None and self._latency_hist is not None:
            with suppress(Exception):
                self._latency_hist.labels(**labels).observe(float(latency_ms) / 1000.0)

    def record_swarm_bid(
        self,
        *,
        agent_id: str,
        task_category: str,
        auction_type: str,
    ) -> None:
        if not self.enabled or self._swarm_bid_counter is None:
            return
        labels = {
            "agent_id": str(agent_id),
            "task_category": str(task_category),
            "auction_type": str(auction_type),
        }
        with suppress(Exception):
            self._swarm_bid_counter.labels(**labels).inc()

    def record_swarm_auction(
        self,
        *,
        auction_type: str,
        task_category: str,
        bid_count: int,
        latency_ms: float | None,
    ) -> None:
        if not self.enabled:
            return
        labels = {
            "task_category": str(task_category),
            "auction_type": str(auction_type),
        }
        if self._swarm_auction_bid_hist is not None:
            with suppress(Exception):
                self._swarm_auction_bid_hist.labels(**labels).observe(int(bid_count))
        if latency_ms is not None and self._swarm_auction_latency_hist is not None:
            with suppress(Exception):
                self._swarm_auction_latency_hist.labels(**labels).observe(
                    float(latency_ms) / 1000.0
                )

    def record_swarm_win(
        self,
        *,
        agent_id: str,
        task_category: str,
        auction_type: str,
        reward: float | None,
        settlement_cost: float | None,
    ) -> None:
        if not self.enabled:
            return
        labels = {
            "agent_id": str(agent_id),
            "task_category": str(task_category),
            "auction_type": str(auction_type),
        }
        if self._swarm_win_counter is not None:
            with suppress(Exception):
                self._swarm_win_counter.labels(**labels).inc()
        if reward is not None and self._swarm_reward_hist is not None:
            with suppress(Exception):
                self._swarm_reward_hist.labels(**labels).observe(float(reward))
        if settlement_cost is not None and self._swarm_settlement_hist is not None:
            with suppress(Exception):
                self._swarm_settlement_hist.labels(**labels).observe(float(settlement_cost))

    def record_span(self, span: Any) -> None:
        if not self.enabled or span is None:
            return
        if getattr(span, "name", None) != "llm.generate":
            return

        attrs = getattr(span, "attributes", {}) or {}
        model = attrs.get("llm.model") or "unknown"
        provider = attrs.get("llm.provider") or "unknown"
        total_tokens = attrs.get("llm.total_tokens")
        if total_tokens is None:
            prompt = attrs.get("llm.prompt_tokens")
            completion = attrs.get("llm.completion_tokens")
            if prompt is not None or completion is not None:
                total_tokens = int(prompt or 0) + int(completion or 0)
        cost_usd = attrs.get("llm.cost_usd")
        latency_ms = attrs.get("llm.latency_ms")

        self.record_llm(
            model=str(model),
            provider=str(provider),
            cost_usd=float(cost_usd) if cost_usd is not None else None,
            total_tokens=int(total_tokens) if total_tokens is not None else None,
            latency_ms=float(latency_ms) if latency_ms is not None else None,
        )

    def record_agent_evolution_cycle(self, *, agent_id: str, proposals: int | None = None) -> None:
        """Record one self-improvement cycle."""
        del proposals
        if not self.enabled or self._agent_evolution_cycles is None:
            return
        with suppress(Exception):
            self._agent_evolution_cycles.labels(agent_id=str(agent_id)).inc()

    def record_agent_evolution_patch(
        self,
        *,
        agent_id: str,
        patch_type: str,
        status: str,
        eval_score: float | None = None,
        rollout_pct: float | None = None,
    ) -> None:
        """Record a proposed, blocked, canaried, promoted, or rolled-back patch."""
        if not self.enabled:
            return
        labels = {
            "agent_id": str(agent_id),
            "patch_type": str(patch_type),
            "status": str(status),
        }
        if self._agent_evolution_patches is not None:
            with suppress(Exception):
                self._agent_evolution_patches.labels(**labels).inc()
        if eval_score is not None and self._agent_evolution_eval_score is not None:
            with suppress(Exception):
                self._agent_evolution_eval_score.labels(**labels).observe(float(eval_score))
        if rollout_pct is not None and self._agent_evolution_rollout_pct is not None:
            with suppress(Exception):
                self._agent_evolution_rollout_pct.labels(**labels).observe(float(rollout_pct))
        if self._agent_evolution_status is not None:
            with suppress(Exception):
                status_code = _EVOLUTION_STATUSES.get(str(status), -1.0)
                self._agent_evolution_status.labels(
                    agent_id=str(agent_id),
                    patch_type=str(patch_type),
                ).observe(status_code)
