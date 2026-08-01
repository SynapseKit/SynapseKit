"""CI-enforced gate for the #734 AgentSwarm market-routing acceptance criteria.

Runs the same 100-task comparison as ``agent_swarm_bench.py`` on a fixed seed and
hard-enforces the issue's two quantitative targets:

* market swarm beats a hardcoded (round-robin) ``AgentFederation`` on mean
  outcome quality by >= 15%;
* market swarm costs >= 25% less than always-routing-to-the-best-model.
"""

from __future__ import annotations

import asyncio

import pytest
from agent_swarm_bench import compare, cost_reduction, quality_improvement

_N_TASKS = 100
_SEED = 42
_MIN_QUALITY_IMPROVEMENT = 0.15
_MIN_COST_REDUCTION = 0.25


@pytest.fixture(scope="module")
def results():
    return asyncio.run(compare(_N_TASKS, _SEED))


def test_swarm_beats_hardcoded_federation_quality_by_15pct(results):
    improvement = quality_improvement(results["swarm"], results["hardcoded_federation"])
    assert improvement >= _MIN_QUALITY_IMPROVEMENT, (
        f"swarm quality={results['swarm'].mean_quality:.3f} vs hardcoded "
        f"federation={results['hardcoded_federation'].mean_quality:.3f} "
        f"improvement={improvement:.1%} < {_MIN_QUALITY_IMPROVEMENT:.0%}"
    )


def test_swarm_costs_25pct_less_than_always_best(results):
    reduction = cost_reduction(results["swarm"], results["always_best"])
    assert reduction >= _MIN_COST_REDUCTION, (
        f"swarm cost={results['swarm'].mean_cost:.2f} vs always-best="
        f"{results['always_best'].mean_cost:.2f} reduction={reduction:.1%} "
        f"< {_MIN_COST_REDUCTION:.0%}"
    )


def test_swarm_quality_at_least_matches_always_best(results):
    # The swarm should not sacrifice quality for its cost win — it matches or
    # beats the premium always-best model while spending far less.
    assert results["swarm"].mean_quality >= results["always_best"].mean_quality


def test_deterministic_under_fixed_seed():
    first = asyncio.run(compare(_N_TASKS, _SEED))
    second = asyncio.run(compare(_N_TASKS, _SEED))
    for name in first:
        assert first[name] == second[name]
