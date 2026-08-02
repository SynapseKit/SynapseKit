"""CI-enforced gate for the #733 neuro-symbolic acceptance criterion.

Runs the arithmetic benchmark and hard-enforces that routing through the symbolic
solver measurably beats answering directly:

* neuro-symbolic accuracy is near-perfect (>= 95%);
* it improves on the LLM-only baseline by >= 25 percentage points;
* the baseline is genuinely fallible (< 85%), so the uplift is real, not trivial.
"""

from __future__ import annotations

import asyncio

import pytest
from neuro_symbolic_bench import compare, uplift

_MIN_NEURO_ACCURACY = 0.95
_MIN_UPLIFT = 0.25
_MAX_BASELINE_ACCURACY = 0.85


@pytest.fixture(scope="module")
def results():
    return asyncio.run(compare())


def test_neuro_symbolic_is_near_perfect(results):
    acc = results["neuro_symbolic"].rate
    assert acc >= _MIN_NEURO_ACCURACY, (
        f"neuro-symbolic accuracy {acc:.0%} < {_MIN_NEURO_ACCURACY:.0%}"
    )


def test_uplift_over_llm_only(results):
    up = uplift(results["neuro_symbolic"], results["llm_only"])
    assert up >= _MIN_UPLIFT, (
        f"neuro-symbolic={results['neuro_symbolic'].rate:.0%} vs "
        f"llm-only={results['llm_only'].rate:.0%} → uplift {up:.0%} < {_MIN_UPLIFT:.0%}"
    )


def test_baseline_is_actually_fallible(results):
    # Guard against a trivially-easy suite: the direct-answer baseline must miss some.
    acc = results["llm_only"].rate
    assert acc <= _MAX_BASELINE_ACCURACY, (
        f"baseline too accurate ({acc:.0%}); suite isn't testing anything"
    )
