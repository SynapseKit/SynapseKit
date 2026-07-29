"""Local Llama 3.2 3B vs. cloud Claude accuracy/latency/cost comparison.

Satisfies the benchmark acceptance criterion of issue #736: run a 200-task
suite (see ``examples/edge_task_suite.py``) through a local GGUF model and a
cloud model **directly** -- not through ``EdgeRuntime`` -- and report where
local is good enough.

The head-to-head, not the routing, is the point: the per-category accuracy gap
is what tells ``CostQualityRouter`` (#589) which task families can stay on the
device. ``EdgeRuntime``'s own routing behaviour is covered by
``tests/llm/test_edge_runtime.py``.

This measures model *quality*, not library wall-clock speed, so it's a
standalone script rather than part of the ``benchmarks/pytest.ini`` timing
suite. See ``test_edge_local_vs_cloud_bench.py`` for the CI-enforced check that
the harness itself is correct.

Usage:
    python benchmarks/edge_local_vs_cloud_bench.py \
        --model-path /models/Llama-3.2-3B-Instruct-Q4_K_M.gguf --n-tasks 200
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from edge_task_suite import CATEGORIES, EdgeTask, generate_tasks, grade

from synapsekit.evaluation.dataset import EvalRecord
from synapsekit.llm.base import BaseLLM, LLMConfig
from synapsekit.observability.tracer import COST_TABLE

LOCAL_COST_USD = 0.0


def _cost_usd(llm: BaseLLM, prev_in: int, prev_out: int) -> float:
    """USD cost of the tokens consumed since the snapshot; 0.0 for local models."""
    pricing = COST_TABLE.get(llm.config.model)
    if not pricing:
        return LOCAL_COST_USD
    delta_in = max(0, llm._input_tokens - prev_in)
    delta_out = max(0, llm._output_tokens - prev_out)
    return delta_in * pricing["input"] + delta_out * pricing["output"]


async def run_task(llm: BaseLLM, task: EdgeTask) -> EvalRecord:
    """Run one task and return its scored record.

    A provider failure scores 0.0 rather than aborting the run -- a model that
    errors on a task is worse at that task, which is exactly what the benchmark
    is measuring.
    """
    prev_in, prev_out = llm._input_tokens, llm._output_tokens
    started = time.perf_counter()
    try:
        output = await llm.generate(task.prompt)
    except Exception as exc:  # noqa: BLE001 - a failed task is a scored 0, not a crash
        output = f"<error: {exc}>"
    latency_ms = (time.perf_counter() - started) * 1000

    return EvalRecord(
        name=task.name,
        score=grade(task, output),
        cost_usd=_cost_usd(llm, prev_in, prev_out),
        latency_ms=latency_ms,
        input=task.prompt,
        output=output,
        ideal=task.expected,
        raw={"category": task.category, "model": llm.config.model},
    )


async def run_suite(llm: BaseLLM, tasks: list[EdgeTask]) -> list[EvalRecord]:
    """Run every task sequentially, so latency samples aren't skewed by contention."""
    return [await run_task(llm, task) for task in tasks]


def summarize(records: list[EvalRecord]) -> dict[str, float]:
    """Aggregate accuracy, latency percentiles, and total cost for one model."""
    scores = [r.score or 0.0 for r in records]
    latencies = sorted(r.latency_ms or 0.0 for r in records)
    return {
        "accuracy": sum(scores) / len(scores) if scores else 0.0,
        "p50_ms": statistics.median(latencies) if latencies else 0.0,
        "p95_ms": latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)]
        if latencies
        else 0.0,
        "total_cost_usd": sum(r.cost_usd or 0.0 for r in records),
    }


def accuracy_by_category(records: list[EvalRecord]) -> dict[str, float]:
    """Mean accuracy per task category."""
    result = {}
    for category in CATEGORIES:
        scores = [
            r.score or 0.0 for r in records if (r.raw or {}).get("category") == category
        ]
        result[category] = sum(scores) / len(scores) if scores else 0.0
    return result


def report(local: list[EvalRecord], cloud: list[EvalRecord]) -> None:
    """Print the comparison table and the per-category gap."""
    local_stats, cloud_stats = summarize(local), summarize(cloud)

    print(f"{'model':<10}{'accuracy':>10}{'p50 ms':>10}{'p95 ms':>10}{'cost USD':>12}")
    for label, stats in (("local", local_stats), ("cloud", cloud_stats)):
        print(
            f"{label:<10}{stats['accuracy']:>10.3f}{stats['p50_ms']:>10.0f}"
            f"{stats['p95_ms']:>10.0f}{stats['total_cost_usd']:>12.4f}"
        )
    print()

    local_cat, cloud_cat = accuracy_by_category(local), accuracy_by_category(cloud)
    print(f"{'category':<16}{'local':>10}{'cloud':>10}{'gap':>10}")
    for category in CATEGORIES:
        gap = local_cat[category] - cloud_cat[category]
        print(
            f"{category:<16}{local_cat[category]:>10.3f}"
            f"{cloud_cat[category]:>10.3f}{gap:>+10.3f}"
        )
    print()
    print("Negative gap = cloud is better; near-zero = local is good enough to keep on-device.")


async def run(
    model_path: str, cloud_model: str, n_tasks: int, seed: int, n_ctx: int
) -> None:
    from synapsekit.llm.anthropic import AnthropicLLM
    from synapsekit.llm.llamacpp import LlamaCppLLM

    tasks = generate_tasks(n=n_tasks, seed=seed)

    local = LlamaCppLLM(
        LLMConfig(model="llama-3.2-3b", api_key="", provider="llamacpp"),
        model_path=model_path,
        n_ctx=n_ctx,
    )
    cloud = AnthropicLLM(
        LLMConfig(
            model=cloud_model,
            api_key=os.environ["ANTHROPIC_API_KEY"],
            provider="anthropic",
        )
    )

    print(f"suite: {len(tasks)} tasks, seed={seed}")
    print(f"local: {model_path}")
    print(f"cloud: {cloud_model}\n")

    local_records = await run_suite(local, tasks)
    cloud_records = await run_suite(cloud, tasks)
    report(local_records, cloud_records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="path to a local GGUF file")
    parser.add_argument("--cloud-model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--n-tasks", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-ctx", type=int, default=4096)
    args = parser.parse_args()
    asyncio.run(
        run(args.model_path, args.cloud_model, args.n_tasks, args.seed, args.n_ctx)
    )


if __name__ == "__main__":
    main()
