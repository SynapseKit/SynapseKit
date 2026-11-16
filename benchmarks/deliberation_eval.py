"""Paired exact-answer evaluation; requires an explicitly supplied adapter factory."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import math
from pathlib import Path

from synapsekit.llm.deliberation import BoundedAdapter, _run, _score, _total, deliberate


def _sum_known(values):
    return None if any(value is None for value in values) else sum(values)


def _validate_config(n, short_cap, long_cap, early_exit_confidence):
    for name, value in (("n", n), ("short_cap", short_cap), ("long_cap", long_cap)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if early_exit_confidence is not None and (
        isinstance(early_exit_confidence, bool)
        or not isinstance(early_exit_confidence, (int, float))
        or not math.isfinite(early_exit_confidence)
        or not 0 <= early_exit_confidence <= 1
    ):
        raise ValueError("early_exit_confidence must be finite and between 0 and 1")


def _validate_cases(cases):
    if not isinstance(cases, list):
        raise ValueError("fixture must be a list of cases")
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or not all(
            isinstance(case.get(field), str) for field in ("prompt", "answer")
        ):
            raise ValueError(f"fixture case {index} must have string prompt and answer")


async def evaluate(
    adapter: BoundedAdapter,
    cases: list[dict],
    *,
    n: int,
    short_cap: int,
    long_cap: int,
    early_exit_confidence: float | None = None,
) -> dict:
    """Score both methods on identical cases, without a cost-incurring judge."""
    _validate_config(n, short_cap, long_cap, early_exit_confidence)
    _validate_cases(cases)
    rows = []
    for case in cases:
        baseline_prompt = (
            "Independently solve the question carefully, explaining your reasoning "
            "step by step before giving the final answer. Return only JSON with "
            '"reasoning" (string), "answer" (string), and "confidence" '
            "(number from 0 to 1).\nQuestion:\n" + case["prompt"]
        )
        baseline = await _run(adapter, baseline_prompt, long_cap, 0)
        baseline_answer, _, _ = _score([baseline])
        parallel = await deliberate(
            adapter,
            case["prompt"],
            n=n,
            max_output_tokens=short_cap,
            total_output_budget=n * short_cap,
            early_exit_confidence=early_exit_confidence,
        )
        expected = case["answer"].strip()
        rows.append(
            {
                "prompt": case["prompt"],
                "expected": expected,
                "baseline_answer": baseline_answer,
                "baseline_trace": baseline.completion.text if baseline.completion else None,
                "baseline_status": baseline.status,
                "parallel_answer": parallel.answer,
                "parallel_early_exit": parallel.early_exit,
                "parallel_launched": len(parallel.traces),
                "parallel_cancelled": sum(t.status == "cancelled" for t in parallel.traces),
                "baseline_correct": baseline_answer == expected,
                "parallel_correct": parallel.answer == expected,
                "baseline_cost": _total([baseline], "cost"),
                "parallel_cost": parallel.total_cost,
                "baseline_input_tokens": _total([baseline], "input_tokens"),
                "parallel_input_tokens": parallel.total_input_tokens,
                "baseline_output_tokens": _total([baseline], "output_tokens"),
                "parallel_output_tokens": parallel.total_output_tokens,
            }
        )
    return {
        "cases": rows,
        "early_exit_confidence": early_exit_confidence,
        "parallel_early_exit_cases": sum(row["parallel_early_exit"] for row in rows),
        "parallel_total_launched": sum(row["parallel_launched"] for row in rows),
        "parallel_total_cancelled": sum(row["parallel_cancelled"] for row in rows),
        "baseline_correct": sum(row["baseline_correct"] for row in rows),
        "parallel_correct": sum(row["parallel_correct"] for row in rows),
        "baseline_total_cost": _sum_known([row["baseline_cost"] for row in rows]),
        "parallel_total_cost": _sum_known([row["parallel_cost"] for row in rows]),
        "baseline_total_input_tokens": _sum_known([row["baseline_input_tokens"] for row in rows]),
        "parallel_total_input_tokens": _sum_known([row["parallel_input_tokens"] for row in rows]),
        "baseline_total_output_tokens": _sum_known([row["baseline_output_tokens"] for row in rows]),
        "parallel_total_output_tokens": _sum_known([row["parallel_output_tokens"] for row in rows]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter", required=True, help="module:zero_argument_factory; may incur charges"
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests/llm/fixtures/deliberation_eval.json",
    )
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--short-cap", type=int, default=128)
    parser.add_argument("--long-cap", type=int, default=1024)
    parser.add_argument("--early-exit-confidence", type=float, default=None)
    args = parser.parse_args()
    try:
        _validate_config(args.n, args.short_cap, args.long_cap, args.early_exit_confidence)
    except ValueError as exc:
        parser.error(str(exc))
    module, name = args.adapter.split(":", 1)
    try:
        cases = json.loads(args.fixture.read_text(encoding="utf-8"))
        _validate_cases(cases)
    except (OSError, ValueError) as exc:
        parser.error(f"invalid fixture: {exc}")
    adapter = getattr(importlib.import_module(module), name)()

    async def run():
        try:
            return await evaluate(
                adapter,
                cases,
                n=args.n,
                short_cap=args.short_cap,
                long_cap=args.long_cap,
                early_exit_confidence=args.early_exit_confidence,
            )
        finally:
            close = getattr(adapter, "aclose", None)
            if close is not None:
                await close()

    report = asyncio.run(run())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
