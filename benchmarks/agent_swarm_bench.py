"""Market-based ``AgentSwarm`` vs. hardcoded orchestration — quality & cost.

Completes the #734 acceptance criteria that had no code against them:

* **quality**: a 5-agent (+1 premium generalist) swarm must beat a *hardcoded*
  ``AgentFederation`` (round-robin — category-agnostic static routing) on a
  100-task benchmark by >=15%.
* **cost**: the swarm must cost >=25% less than an *always-route-to-best-model*
  baseline (every task to the single highest-average-quality agent).

The scenario is emergent specialization made measurable: each specialist agent
is expert (high quality) in exactly one task category and mediocre elsewhere,
at a modest cost. A premium generalist is good everywhere but expensive. The
market routes each task to the in-category specialist (high quality, low cost);
hardcoded round-robin can't adapt to category, and always-best overpays for the
generalist. Deterministic given ``seed`` (``MarketPolicy.seed`` + a seeded task
stream, ``exploration_rate=0``).

This measures routing *quality/cost*, not wall-clock, so it's a standalone
script. See ``test_agent_swarm_bench.py`` for the CI-enforced gate.

Usage:
    python benchmarks/agent_swarm_bench.py --n-tasks 100 --seed 42
"""

from __future__ import annotations

import argparse
import asyncio
import random
from dataclasses import dataclass

from synapsekit.agents import (
    AgentFederation,
    AgentMetadata,
    AgentSwarm,
    Bid,
    InMemoryAgentRegistry,
    MarketPolicy,
)

CATEGORIES = ("research", "summarize", "code", "translate", "math")

_SPECIALIST_QUALITY = 0.95
_OFF_SPECIALTY_QUALITY = 0.55
_SPECIALIST_COST = 10.0
_PREMIUM_QUALITY = 0.90
_PREMIUM_COST = 40.0
_PREMIUM_ID = "premium"


@dataclass(frozen=True)
class AgentSpec:
    """A benchmark agent's latent per-category quality and flat cost."""

    id: str
    quality_by_category: dict[str, float]
    default_quality: float
    cost: float

    def quality(self, category: str) -> float:
        return self.quality_by_category.get(category, self.default_quality)


def build_agents() -> list[AgentSpec]:
    """One specialist per category + a premium generalist (highest avg quality)."""
    specs: list[AgentSpec] = [
        AgentSpec(
            id=f"specialist_{category}",
            quality_by_category={category: _SPECIALIST_QUALITY},
            default_quality=_OFF_SPECIALTY_QUALITY,
            cost=_SPECIALIST_COST,
        )
        for category in CATEGORIES
    ]
    specs.append(
        AgentSpec(
            id=_PREMIUM_ID,
            quality_by_category={},
            default_quality=_PREMIUM_QUALITY,
            cost=_PREMIUM_COST,
        )
    )
    return specs


def best_average_agent(specs: list[AgentSpec]) -> str:
    """The single agent you'd pick for everything — highest mean quality."""
    return max(specs, key=lambda s: sum(s.quality(c) for c in CATEGORIES) / len(CATEGORIES)).id


def make_tasks(n_tasks: int, seed: int) -> list[str]:
    """A seeded stream of ``"<category>::task-<i>"`` prompts (category encoded)."""
    rng = random.Random(seed)
    return [f"{rng.choice(CATEGORIES)}::task-{i}" for i in range(n_tasks)]


def _category_of(prompt: str) -> str:
    return prompt.split("::", 1)[0]


class BenchClient:
    """Agent client that bids and returns its true latent outcome per category."""

    def __init__(self, spec: AgentSpec) -> None:
        self.spec = spec

    def bid(self, task: str, *, task_category: str | None = None, **_: object) -> Bid:
        category = task_category or _category_of(task)
        return Bid(
            agent_id=self.spec.id,
            estimated_cost=self.spec.cost,
            estimated_quality=self.spec.quality(category),
            confidence=0.9,
            task_category=category,
        )

    async def run(self, prompt: str, **_: object) -> dict[str, object]:
        category = _category_of(prompt)
        return {
            "agent_id": self.spec.id,
            "actual_quality": self.spec.quality(category),
            "actual_cost": self.spec.cost,
        }


def _registry(specs: list[AgentSpec]) -> tuple[InMemoryAgentRegistry, dict[str, BenchClient]]:
    registry = InMemoryAgentRegistry()
    clients: dict[str, BenchClient] = {}
    for spec in specs:
        registry.register(AgentMetadata(id=spec.id, model="bench", capacity=1_000_000))
        clients[spec.id] = BenchClient(spec)
    return registry, clients


@dataclass(frozen=True)
class StrategyResult:
    mean_quality: float
    mean_cost: float


async def run_swarm(specs: list[AgentSpec], tasks: list[str], seed: int) -> StrategyResult:
    registry, clients = _registry(specs)
    swarm = AgentSwarm(
        market=MarketPolicy(seed=seed, exploration_rate=0.0, budget_per_task=10_000.0),
        registry=registry,
        clients=clients,
    )
    qualities, costs = [], []
    for prompt in tasks:
        result = await swarm.execute(prompt, task_category=_category_of(prompt))
        qualities.append(result.actual_quality)
        costs.append(result.actual_cost)
    return StrategyResult(_mean(qualities), _mean(costs))


async def run_hardcoded_federation(specs: list[AgentSpec], tasks: list[str]) -> StrategyResult:
    """Baseline: real ``AgentFederation`` round-robin (category-agnostic)."""
    registry, clients = _registry(specs)
    federation = AgentFederation(registry, clients=clients)
    qualities, costs = [], []
    for prompt in tasks:
        output = await federation.run(prompt)
        qualities.append(float(output["actual_quality"]))
        costs.append(float(output["actual_cost"]))
    return StrategyResult(_mean(qualities), _mean(costs))


async def run_always_best(specs: list[AgentSpec], tasks: list[str]) -> StrategyResult:
    """Baseline: every task routed to the single best-average-quality agent."""
    registry, clients = _registry(specs)
    federation = AgentFederation(registry, clients=clients)
    best_id = best_average_agent(specs)
    qualities, costs = [], []
    for prompt in tasks:
        output = await federation.run(prompt, agent_id=best_id)
        qualities.append(float(output["actual_quality"]))
        costs.append(float(output["actual_cost"]))
    return StrategyResult(_mean(qualities), _mean(costs))


async def compare(n_tasks: int, seed: int) -> dict[str, StrategyResult]:
    specs = build_agents()
    tasks = make_tasks(n_tasks, seed)
    swarm, federation, always_best = await asyncio.gather(
        run_swarm(specs, tasks, seed),
        run_hardcoded_federation(specs, tasks),
        run_always_best(specs, tasks),
    )
    return {"swarm": swarm, "hardcoded_federation": federation, "always_best": always_best}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def quality_improvement(swarm: StrategyResult, baseline: StrategyResult) -> float:
    if baseline.mean_quality == 0:
        return float("inf") if swarm.mean_quality > 0 else 0.0
    return (swarm.mean_quality - baseline.mean_quality) / baseline.mean_quality


def cost_reduction(swarm: StrategyResult, baseline: StrategyResult) -> float:
    if baseline.mean_cost == 0:
        return 0.0
    return (baseline.mean_cost - swarm.mean_cost) / baseline.mean_cost


async def run(n_tasks: int, seed: int) -> None:
    results = await compare(n_tasks, seed)
    print(f"tasks: {n_tasks}, seed: {seed}, categories: {len(CATEGORIES)}\n")
    print(f"{'strategy':<24}{'mean_quality':>14}{'mean_cost':>12}")
    for name in ("swarm", "hardcoded_federation", "always_best"):
        r = results[name]
        print(f"{name:<24}{r.mean_quality:>14.3f}{r.mean_cost:>12.2f}")
    print()
    q = quality_improvement(results["swarm"], results["hardcoded_federation"])
    c = cost_reduction(results["swarm"], results["always_best"])
    print(f"swarm quality improvement vs hardcoded federation: {q:+.1%} (target >= +15%)")
    print(f"swarm cost reduction vs always-best-model:         {c:+.1%} (target >= +25%)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-tasks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    asyncio.run(run(args.n_tasks, args.seed))


if __name__ == "__main__":
    main()
