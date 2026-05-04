"""``synapsekit benchmark`` command: run agent benchmarks."""

from __future__ import annotations

import importlib
from typing import Any

from ..evaluation.benchmarks import BENCHMARK_REGISTRY


def run_benchmark(args: Any) -> None:
    """Run an agent benchmark."""
    subcommand = getattr(args, "benchmark_command", None)

    if subcommand == "run":
        _run_benchmark_suite(args)
        return

    if subcommand == "list":
        _run_list_benchmarks(args)
        return

    raise SystemExit("Missing benchmark subcommand. Use: run or list")


def _run_list_benchmarks(args: Any) -> None:
    """List available benchmarks."""
    print("Available Benchmarks:")
    for key, cls in BENCHMARK_REGISTRY.items():
        print(f"  - {key} ({cls.name})")


def _run_benchmark_suite(args: Any) -> None:
    """Run a specific benchmark."""
    benchmark_key = args.suite
    if benchmark_key not in BENCHMARK_REGISTRY:
        raise SystemExit(f"Unknown benchmark suite: {benchmark_key}")

    # Dynamically load the agent
    agent_path = args.agent
    try:
        module_name, func_name = agent_path.split(":", 1)
        module = importlib.import_module(module_name)
        agent = getattr(module, func_name)
    except Exception as e:
        raise SystemExit(f"Failed to load agent '{agent_path}': {e}") from e

    benchmark_cls = BENCHMARK_REGISTRY[benchmark_key]
    benchmark = benchmark_cls()

    print(f"Running benchmark {benchmark.name}...")
    result = benchmark.evaluate(agent, split=args.split, limit=args.limit)

    print("\n" + result.format_leaderboard())
