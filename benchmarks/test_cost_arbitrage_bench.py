"""CI gate for the issue #893 cost-arbitrage acceptance criteria."""

from __future__ import annotations

from cost_arbitrage_bench import run_simulation

_N_TASKS = 100
_SEED = 42
_MIN_SAVINGS = 0.40


def test_replay_is_deterministic() -> None:
    assert run_simulation(_N_TASKS, _SEED) == run_simulation(_N_TASKS, _SEED)


def test_arbitrage_saves_at_least_40pct_at_equal_quality() -> None:
    result = run_simulation(_N_TASKS, _SEED)
    assert result.tasks == _N_TASKS
    assert result.equal_quality
    assert result.savings_pct >= _MIN_SAVINGS, (
        f"arbitrage savings={result.savings_pct:.1%} < target={_MIN_SAVINGS:.0%}; "
        f"routed=${result.routed_cost_usd:.6f}, baseline=${result.baseline_cost_usd:.6f}"
    )
