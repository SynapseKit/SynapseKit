"""SynapseKit CLI — ``synapsekit serve`` and ``synapsekit test``."""

from __future__ import annotations

import argparse
import sys


def _add_serve_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("serve", help="Serve a SynapseKit app as a FastAPI server")
    p.add_argument("app", help="Import path, e.g. 'my_module:rag'")
    p.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    p.add_argument("--reload", action="store_true", help="Enable auto-reload")


def _add_test_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("test", help="Run evaluation test suites")
    p.add_argument("path", nargs="?", default=".", help="Directory or file to scan (default: .)")
    p.add_argument(
        "--threshold", type=float, default=0.7, help="Min score threshold (default: 0.7)"
    )
    p.add_argument(
        "--format",
        dest="output_format",
        choices=["json", "table"],
        default="table",
        help="Output format (default: table)",
    )
    p.add_argument(
        "--save",
        dest="save_snapshot",
        metavar="NAME",
        help="Save results as a named snapshot",
    )
    p.add_argument(
        "--compare",
        dest="compare_baseline",
        metavar="BASELINE",
        help="Compare results against a saved baseline snapshot",
    )
    p.add_argument(
        "--fail-on-regression",
        action="store_true",
        default=False,
        help="Exit with code 1 if regressions are detected",
    )
    p.add_argument(
        "--snapshot-dir",
        default=".synapsekit_evals",
        help="Snapshot storage directory (default: .synapsekit_evals)",
    )


def _add_eval_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("eval", help="EvalCI snapshot report/export/compare")
    eval_sub = p.add_subparsers(dest="eval_command")

    report = eval_sub.add_parser("report", help="Summarize a saved eval snapshot")
    report.add_argument("snapshot", help="Snapshot name")
    report.add_argument("--threshold", type=float, default=0.8, help="Weak-case threshold")
    report.add_argument("--snapshot-dir", default=".synapsekit_evals", help="Snapshot storage dir")

    export = eval_sub.add_parser("export", help="Export snapshot to fine-tune dataset")
    export.add_argument("snapshot", help="Snapshot name")
    export.add_argument(
        "--format",
        choices=["openai", "anthropic", "together", "jsonl", "dpo"],
        default="openai",
        help="Export format",
    )
    export.add_argument("--min-score", type=float, default=None, help="Minimum score filter")
    export.add_argument("--max-score", type=float, default=None, help="Maximum score filter")
    export.add_argument("--output", required=True, help="Output JSONL path")
    export.add_argument("--snapshot-dir", default=".synapsekit_evals", help="Snapshot storage dir")

    compare = eval_sub.add_parser("compare", help="Compare two saved eval snapshots")
    compare.add_argument("baseline", help="Baseline snapshot")
    compare.add_argument("current", help="Current snapshot")
    compare.add_argument("--snapshot-dir", default=".synapsekit_evals", help="Snapshot storage dir")


def _add_finetune_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("finetune", help="Submit and monitor fine-tuning jobs")
    ft_sub = p.add_subparsers(dest="finetune_command")

    submit = ft_sub.add_parser("submit", help="Submit fine-tuning job")
    submit.add_argument("dataset", help="Dataset file ID/path accepted by provider")
    submit.add_argument("--provider", choices=["openai", "together"], required=True)
    submit.add_argument("--base-model", required=True, help="Base model name")
    submit.add_argument("--job-name", default=None, help="Optional job suffix/name")
    submit.add_argument("--n-epochs", type=int, default=3, help="Training epochs")
    submit.add_argument("--api-key", default=None, help="Provider API key")

    status = ft_sub.add_parser("status", help="Get fine-tune job status")
    status.add_argument("job_id", help="Fine-tune job ID")
    status.add_argument("--provider", choices=["openai", "together"], required=True)
    status.add_argument("--api-key", default=None, help="Provider API key")

    wait = ft_sub.add_parser("wait", help="Wait for fine-tune job completion")
    wait.add_argument("job_id", help="Fine-tune job ID")
    wait.add_argument("--provider", choices=["openai", "together"], required=True)
    wait.add_argument("--interval", type=float, default=10.0, help="Poll interval seconds")
    wait.add_argument("--timeout", type=float, default=3600.0, help="Timeout seconds")
    wait.add_argument("--api-key", default=None, help="Provider API key")


def _add_graph_builder_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("graph-builder", help="Launch the visual graph workflow builder")
    p.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=7861, help="Bind port (default: 7861)")


def _add_benchmark_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("benchmark", help="Run agent benchmarks (GAIA, SWE-bench, etc.)")
    bench_sub = p.add_subparsers(dest="benchmark_command")

    run_cmd = bench_sub.add_parser("run", help="Run a specific benchmark suite")
    run_cmd.add_argument("suite", help="Benchmark suite (gaia, swe-bench, webarena, agentbench)")
    run_cmd.add_argument("agent", help="Import path for the agent, e.g. 'my_agent:run_task'")
    run_cmd.add_argument("--split", default="test", help="Dataset split to evaluate on")
    run_cmd.add_argument("--limit", type=int, default=None, help="Max tasks to evaluate")

    bench_sub.add_parser("list", help="List available benchmarks")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="synapsekit",
        description="SynapseKit CLI — serve apps and run evaluations",
    )
    parser.add_argument("--version", action="store_true", help="Show version and exit")

    subparsers = parser.add_subparsers(dest="command")
    _add_serve_parser(subparsers)
    _add_test_parser(subparsers)
    _add_eval_parser(subparsers)
    _add_finetune_parser(subparsers)
    _add_graph_builder_parser(subparsers)
    _add_benchmark_parser(subparsers)

    args = parser.parse_args(argv)

    if args.version:
        from synapsekit import __version__

        print(f"synapsekit {__version__}")
        return

    if args.command == "serve":
        from .serve import run_serve

        run_serve(args)
    elif args.command == "test":
        from .test import run_test

        run_test(args)
    elif args.command == "eval":
        from .eval import run_eval

        run_eval(args)
    elif args.command == "finetune":
        from .finetune import run_finetune

        run_finetune(args)
    elif args.command == "graph-builder":
        from .graph_builder import run_graph_builder

        run_graph_builder(args)
    elif args.command == "benchmark":
        from .benchmark import run_benchmark

        run_benchmark(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
