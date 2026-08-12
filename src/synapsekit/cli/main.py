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


def _add_ui_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("ui", help="Launch the observability dashboard")
    p.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=7860, help="Bind port (default: 7860)")
    p.add_argument(
        "--live",
        action="store_true",
        help="Lightweight zero-dependency live run dashboard (stdlib only, no extras needed)",
    )


def _add_plugin_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("plugin", help="Manage SynapseKit plugins")
    plugin_sub = p.add_subparsers(dest="plugin_command")

    plugin_sub.add_parser("list", help="List all registered plugins")

    load_cmd = plugin_sub.add_parser("load", help="Load a plugin from a Python file")
    load_cmd.add_argument("path", help="Path to the plugin .py file")

    info_cmd = plugin_sub.add_parser("info", help="Show details about a registered plugin")
    info_cmd.add_argument("name", help="Plugin name")


def _add_benchmark_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("benchmark", help="Run agent benchmarks (GAIA, SWE-bench, etc.)")
    bench_sub = p.add_subparsers(dest="benchmark_command")

    run_cmd = bench_sub.add_parser("run", help="Run a specific benchmark suite")
    run_cmd.add_argument("suite", help="Benchmark suite (gaia, swe-bench, webarena, agentbench)")
    run_cmd.add_argument("agent", help="Import path for the agent, e.g. 'my_agent:run_task'")
    run_cmd.add_argument("--split", default="test", help="Dataset split to evaluate on")
    run_cmd.add_argument("--limit", type=int, default=None, help="Max tasks to evaluate")

    bench_sub.add_parser("list", help="List available benchmarks")


def _add_bench_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    from .bench import build_bench_parser

    build_bench_parser(subparsers)


def _add_edge_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    from .edge import build_edge_parser

    build_edge_parser(subparsers)


def _add_audit_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    from .audit import build_audit_parser

    build_audit_parser(subparsers)


def _add_mesh_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    from .mesh import build_mesh_parser

    build_mesh_parser(subparsers)


def _add_shell_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    from .shell import build_shell_parser

    build_shell_parser(subparsers)


def _add_agent_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("agent", help="Inspect and manage SynapseKit agents")
    agent_sub = p.add_subparsers(dest="agent_command")

    inspect_cmd = agent_sub.add_parser(
        "inspect-evolution",
        help="Inspect self-improving agent evolution history",
    )
    inspect_cmd.add_argument("agent_id", help="Agent id to inspect, or 'all'")
    inspect_cmd.add_argument(
        "--audit-path",
        default=".synapsekit_agent_evolution.jsonl",
        help="Evolution audit JSONL path",
    )
    inspect_cmd.add_argument("--limit", type=int, default=20, help="Maximum patches to show")
    inspect_cmd.add_argument(
        "--format",
        dest="output_format",
        choices=["table", "json"],
        default="table",
        help="Output format",
    )

    keygen = agent_sub.add_parser("keygen", help="Generate an Ed25519 publisher key")
    keygen.add_argument("private_key", help="New raw private-key file")
    keygen.add_argument("--public-key", default=None, help="Optional base64 public-key output")
    keygen.add_argument("--key-id", default=None, help="Optional stable publisher key id")

    pack = agent_sub.add_parser("pack", help="Pack and sign a portable .agent bundle")
    pack.add_argument("source", help="Agent source directory")
    pack.add_argument("--output", required=True, help="Output .agent path")
    pack.add_argument("--name", required=True, help="Portable agent name")
    pack.add_argument("--agent-version", required=True, help="Portable agent version")
    pack.add_argument("--author", required=True, help="Publisher or author name")
    pack.add_argument("--private-key", required=True, help="Raw Ed25519 private-key file")
    pack.add_argument("--key-id", default=None, help="Publisher key id")
    pack.add_argument("--description", default="", help="Agent description")
    pack.add_argument("--entrypoint", default=None, help="Optional bundled FILE:SYMBOL")
    pack.add_argument("--tag", dest="tags", action="append", default=[], help="Agent tag")
    pack.add_argument("--eval-score", type=float, default=None, help="Eval score from 0 to 1")

    def add_trust_options(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--trusted-key",
            dest="trusted_keys",
            action="append",
            default=None,
            metavar="KEY_ID:BASE64_PUBLIC_KEY",
            help="Pin an independently obtained publisher key (repeatable)",
        )
        command.add_argument(
            "--require-trusted",
            action="store_true",
            help="Reject bundles whose publisher key is not pinned",
        )

    verify = agent_sub.add_parser("verify", help="Verify hashes, signature, and publisher trust")
    verify.add_argument("bundle", help="Path to a .agent bundle")
    verify.add_argument("--format", dest="output_format", choices=["text", "json"], default="text")
    add_trust_options(verify)

    unpack = agent_sub.add_parser("unpack", help="Verify and unpack a .agent bundle")
    unpack.add_argument("bundle", help="Path to a .agent bundle")
    unpack.add_argument("output", help="New destination directory")
    add_trust_options(unpack)

    install = agent_sub.add_parser("install", help="Verify and install an inert agent bundle")
    install.add_argument("bundle", help="Path to a .agent bundle")
    install.add_argument("--install-root", default=None, help="Agent installation root")
    add_trust_options(install)

    publish = agent_sub.add_parser("publish", help="Publish a bundle to a file-backed registry")
    publish.add_argument("bundle", help="Path to a .agent bundle")
    publish.add_argument("--registry", required=True, help="Registry root directory")
    publish.add_argument(
        "--allow-untrusted",
        action="store_true",
        help="Explicitly allow self-signed publishers in this registry",
    )
    add_trust_options(publish)


def _add_memory_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("memory", help="Manage Living Memory patches")
    mem_sub = p.add_subparsers(dest="memory_command")

    review = mem_sub.add_parser("review", help="Review pending memory patches")
    review.add_argument("--patch-id", default=None, help="Review a specific patch")
    review.add_argument(
        "--store-path",
        default=".synapsekit_memory_patches.jsonl",
        help="Patch store file path",
    )

    apply_cmd = mem_sub.add_parser("apply", help="Apply a pending memory patch")
    apply_cmd.add_argument("patch_id", help="Patch ID to apply")
    apply_cmd.add_argument(
        "--store-path",
        default=".synapsekit_memory_patches.jsonl",
        help="Patch store file path",
    )

    revert_cmd = mem_sub.add_parser("revert", help="Revert an applied memory patch")
    revert_cmd.add_argument("patch_id", help="Patch ID to revert")
    revert_cmd.add_argument(
        "--store-path",
        default=".synapsekit_memory_patches.jsonl",
        help="Patch store file path",
    )

    log_cmd = mem_sub.add_parser("log", help="View memory patch history")
    log_cmd.add_argument("--status", default=None, help="Filter by status")
    log_cmd.add_argument("--limit", type=int, default=20, help="Max patches to show")
    log_cmd.add_argument(
        "--format",
        dest="output_format",
        choices=["table", "json"],
        default="table",
        help="Output format",
    )
    log_cmd.add_argument(
        "--store-path",
        default=".synapsekit_memory_patches.jsonl",
        help="Patch store file path",
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    from .._loop import install_fast_loop

    install_fast_loop()

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
    _add_bench_parser(subparsers)
    _add_edge_parser(subparsers)
    _add_mesh_parser(subparsers)
    _add_shell_parser(subparsers)
    _add_agent_parser(subparsers)
    from .hive import build_hive_parser

    build_hive_parser(subparsers)
    _add_memory_parser(subparsers)
    _add_ui_parser(subparsers)
    _add_plugin_parser(subparsers)
    _add_audit_parser(subparsers)

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
    elif args.command == "bench":
        from .bench import run_bench

        run_bench(args)
    elif args.command == "edge":
        from .edge import run_edge

        run_edge(args)
    elif args.command == "mesh":
        from .mesh import run_mesh

        run_mesh(args)
    elif args.command == "shell":
        from .shell import run_shell

        run_shell(args)
    elif args.command == "agent":
        from .agent import run_agent

        run_agent(args)
    elif args.command == "hive":
        from .hive import run_hive

        run_hive(args)
    elif args.command == "memory":
        from .memory import run_memory

        run_memory(args)
    elif args.command == "ui":
        from .ui import run_ui

        run_ui(args)
    elif args.command == "plugin":
        from .plugins import run_plugin

        run_plugin(args)
    elif args.command == "audit":
        from .audit import run_audit

        run_audit(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
