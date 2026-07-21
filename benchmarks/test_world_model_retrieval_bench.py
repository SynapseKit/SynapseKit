"""CI-enforced regression check: hybrid retrieval must beat vector-only.

Runs the same comparison as ``world_model_retrieval_bench.py`` on a small,
fixed, deterministic corpus so it's fast and reproducible in CI. Asserts an
*absolute* hit-rate gap (not a ratio) because the synthetic question set is
designed so vector-only retrieval scores exactly 0.0 -- the tail (incident)
document shares no vocabulary with a question naming only the head (person)
entity, so it's only reachable by graph traversal. A ratio against a 0.0
baseline is degenerate (always "infinite" improvement); an absolute
percentage-point gap is the well-defined, non-flaky way to hard-enforce the
issue's ">=20% improvement" acceptance criterion here.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from world_model_corpus import generate_corpus
from world_model_retrieval_bench import build_world_model, compare

_N_DOCS = 300
_SEED = 7
_TOP_K = 5
_MIN_ABSOLUTE_IMPROVEMENT = 0.20


@pytest.fixture(scope="module")
def bench_results() -> dict[str, dict[str, float]]:
    async def _run() -> dict[str, dict[str, float]]:
        docs, chains = generate_corpus(n_docs=_N_DOCS, seed=_SEED)
        wm = await build_world_model(docs)
        return await compare(wm, chains, _TOP_K)

    return asyncio.run(_run())


def test_hybrid_hit_rate_beats_vector_only_by_at_least_20_points(bench_results):
    gap = bench_results["hybrid"]["hit_rate"] - bench_results["vector_only"]["hit_rate"]
    assert gap >= _MIN_ABSOLUTE_IMPROVEMENT, (
        f"hybrid hit@{_TOP_K}={bench_results['hybrid']['hit_rate']:.3f} did not beat "
        f"vector-only={bench_results['vector_only']['hit_rate']:.3f} by >= "
        f"{_MIN_ABSOLUTE_IMPROVEMENT:.0%} (gap={gap:.3f})"
    )


def test_hybrid_mrr_beats_vector_only_by_at_least_20_points(bench_results):
    gap = bench_results["hybrid"]["mrr"] - bench_results["vector_only"]["mrr"]
    assert gap >= _MIN_ABSOLUTE_IMPROVEMENT, (
        f"hybrid mrr={bench_results['hybrid']['mrr']:.3f} did not beat "
        f"vector-only={bench_results['vector_only']['mrr']:.3f} by >= "
        f"{_MIN_ABSOLUTE_IMPROVEMENT:.0%} (gap={gap:.3f})"
    )
