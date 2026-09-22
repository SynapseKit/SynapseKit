"""Deterministic provider cost-arbitrage simulation for issue #893.

The harness uses no provider SDKs or network calls. It replays a trace with a
quality floor, routes each request to the cheapest eligible model, and compares
that spend with always using the best-quality model.

Usage:
    uv run python benchmarks/cost_arbitrage_bench.py --n-tasks 100 --seed 42
"""

from __future__ import annotations

import argparse
import random

from synapsekit.llm.finops import (
    CostArbitrageSimulator,
    ModelPricing,
    SimulationModel,
    SimulationTask,
)

_REQUEST_CLASSES = ("chat", "research", "code", "summarize")
_N_INPUT_TOKENS = 600
_N_OUTPUT_TOKENS = 300


def build_models() -> list[SimulationModel]:
    """Return a cheap and premium endpoint with equal measured quality."""
    quality = {request_class: 0.92 for request_class in _REQUEST_CLASSES}
    latency = {request_class: 80.0 for request_class in _REQUEST_CLASSES}
    return [
        SimulationModel(
            provider="economy-provider",
            model="economy-model",
            pricing=ModelPricing(
                provider="economy-provider",
                model="economy-model",
                input_cost_per_million=0.20,
                output_cost_per_million=0.80,
                latency_sla_ms=100.0,
                quality_score=0.92,
            ),
            quality_by_class=quality,
            latency_by_class_ms=latency,
        ),
        SimulationModel(
            provider="premium-provider",
            model="premium-model",
            pricing=ModelPricing(
                provider="premium-provider",
                model="premium-model",
                input_cost_per_million=5.0,
                output_cost_per_million=15.0,
                latency_sla_ms=100.0,
                quality_score=0.92,
            ),
            quality_by_class=quality,
            latency_by_class_ms=latency,
        ),
    ]


def make_trace(n_tasks: int = 100, seed: int = 42) -> list[SimulationTask]:
    rng = random.Random(seed)
    return [
        SimulationTask(
            task_id=f"task-{index}",
            request_class=rng.choice(_REQUEST_CLASSES),
            input_tokens=_N_INPUT_TOKENS,
            output_tokens=_N_OUTPUT_TOKENS,
        )
        for index in range(n_tasks)
    ]


def run_simulation(n_tasks: int = 100, seed: int = 42):
    simulator = CostArbitrageSimulator(build_models())
    return simulator.replay(make_trace(n_tasks, seed), quality_floor=0.90, max_latency_ms=120.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-tasks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_simulation(args.n_tasks, args.seed)
    print(f"tasks: {result.tasks}, seed: {args.seed}")
    print(f"arbitrage cost:      ${result.routed_cost_usd:.6f}")
    print(f"always-best cost:    ${result.baseline_cost_usd:.6f}")
    print(f"saved:               ${result.saved_usd:.6f} ({result.savings_pct:.1%})")
    print(f"quality:             {result.routed_quality:.3f}")
    print(f"baseline quality:    {result.baseline_quality:.3f}")
    print(f"equal quality:       {result.equal_quality}")


if __name__ == "__main__":
    main()
