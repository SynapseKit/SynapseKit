"""SynapseKit CLI entry point."""

from __future__ import annotations

import argparse


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
    p.add_argument("--format", dest="output_format", choices=["json", "table"], default="table")
    p.add_argument("--save", dest="save_snapshot", metavar="NAME")
    p.add_argument("--compare", dest="compare_baseline", metavar="BASELINE")
    p.add_argument("--fail-on-regression", action="store_true", default=False)
    p.add_argument("--snapshot-dir", default=".synapsekit_evals")


def _add_eval_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("eval", help="EvalCI snapshot report/export/compare")
    eval_sub = p.add_subparsers(dest="eval_command")
    report = eval_sub.add_parser("report", help="Summarize a saved eval snapshot")
    report.add_argument("snapshot")
    report.add_argument("--threshold", type=float, default=0.8)
    report.add_argument("--snapshot-dir", default=".synapsekit_evals")
    export = eval_sub.add_parser("export", help="Export snapshot to fine-tune dataset")
    export.add_argument("snapshot")
    export.add_argument(
        "--format", choices=["openai", "anthropic", "together", "jsonl", "dpo"], default="openai"
    )
    export.add_argument("--min-score", type=float, default=None)
    export.add_argument("--max-score", type=float, default=None)
    export.add_argument("--output", required=True)
    export.add_argument("--snapshot-dir", default=".synapsekit_evals")
    compare = eval_sub.add_parser("compare", help="Compare two saved eval snapshots")
    compare.add_argument("baseline")
    compare.add_argument("current")
    compare.add_argument("--snapshot-dir", default=".synapsekit_evals")


def _add_finetune_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("finetune", help="Submit and monitor fine-tuning jobs")
    ft_sub = p.add_subparsers(dest="finetune_command")
    submit = ft_sub.add_parser("submit", help="Submit fine-tuning job")
    submit.add_argument("dataset")
    submit.add_argument("--provider", choices=["openai", "together"], required=True)
    submit.add_argument("--base-model", required=True)
    submit.add_argument("--job-name", default=None)
    submit.add_argument("--n-epochs", type=int, default=3)
    submit.add_argument("--api-key", default=None)
    status = ft_sub.add_parser("status", help="Get fine-tune job status")
    status.add_argument("job_id")
    status.add_argument("--provider", choices=["openai", "together"], required=True)
    status.add_argument("--api-key", default=None)
    wait = ft_sub.add_parser("wait", help="Wait for fine-tune job completion")
    wait.add_argument("job_id")
    wait.add_argument("--provider", choices=["openai", "together"], required=True)
    wait.add_argument("--interval", type=float, default=10.0)
    wait.add_argument("--timeout", type=float, default=3600.0)
    wait.add_argument("--api-key", default=None)


def _add_graph_builder_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("graph-builder", help="Launch the visual graph workflow builder")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7861)


def _add_ui_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("ui", help="Launch the observability dashboard")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--live", action="store_true")


def _add_plugin_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("plugin", help="Manage SynapseKit plugins")
    plugin_sub = p.add_subparsers(dest="plugin_command")
    plugin_sub.add_parser("list", help="List all registered plugins")
    load_cmd = plugin_sub.add_parser("load", help="Load a plugin from a Python file")
    load_cmd.add_argument("path")
    info_cmd = plugin_sub.add_parser("info", help="Show details about a registered plugin")
    info_cmd.add_argument("name")


def _add_benchmark_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("benchmark", help="Run agent benchmarks (GAIA, SWE-bench, etc.)")
    bench_sub = p.add_subparsers(dest="benchmark_command")
    run_cmd = bench_sub.add_parser("run", help="Run a specific benchmark suite")
    run_cmd.add_argument("suite")
    run_cmd.add_argument("agent")
    run_cmd.add_argument("--split", default="test")
    run_cmd.add_argument("--limit", type=int, default=None)
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


def _add_dream_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    from .dream import build_dream_parser

    build_dream_parser(subparsers)


def _add_ambient_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    from .ambient import build_ambient_parser

    build_ambient_parser(subparsers)


def _add_sandbox_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    from .sandbox import build_sandbox_parser

    build_sandbox_parser(subparsers)


def _add_agent_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("agent", help="Inspect and manage SynapseKit agents")
    agent_sub = p.add_subparsers(dest="agent_command")
    inspect_cmd = agent_sub.add_parser(
        "inspect-evolution", help="Inspect self-improving agent evolution history"
    )
    inspect_cmd.add_argument("agent_id")
    inspect_cmd.add_argument("--audit-path", default=".synapsekit_agent_evolution.jsonl")
    inspect_cmd.add_argument("--limit", type=int, default=20)
    inspect_cmd.add_argument(
        "--format", dest="output_format", choices=["table", "json"], default="table"
    )
    keygen = agent_sub.add_parser("keygen", help="Generate an Ed25519 publisher key")
    keygen.add_argument("private_key")
    keygen.add_argument("--public-key", default=None)
    keygen.add_argument("--key-id", default=None)
    pack = agent_sub.add_parser("pack", help="Pack and sign a portable .agent bundle")
    pack.add_argument("source")
    pack.add_argument("--output", required=True)
    pack.add_argument("--name", required=True)
    pack.add_argument("--agent-version", required=True)
    pack.add_argument("--author", required=True)
    pack.add_argument("--private-key", required=True)
    pack.add_argument("--key-id", default=None)
    pack.add_argument("--description", default="")
    pack.add_argument("--entrypoint", default=None)
    pack.add_argument("--tag", dest="tags", action="append", default=[])
    pack.add_argument("--eval-score", type=float, default=None)

    def add_trust_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("--trusted-key", dest="trusted_keys", action="append", default=None)
        command.add_argument("--require-trusted", action="store_true")

    verify = agent_sub.add_parser("verify", help="Verify hashes, signature, and publisher trust")
    verify.add_argument("bundle")
    verify.add_argument("--format", dest="output_format", choices=["text", "json"], default="text")
    add_trust_options(verify)
    unpack = agent_sub.add_parser("unpack", help="Verify and unpack a .agent bundle")
    unpack.add_argument("bundle")
    unpack.add_argument("output")
    add_trust_options(unpack)
    install = agent_sub.add_parser("install", help="Verify and install an inert agent bundle")
    install.add_argument("bundle")
    install.add_argument("--install-root", default=None)
    add_trust_options(install)
    publish = agent_sub.add_parser("publish", help="Publish a bundle to a file-backed registry")
    publish.add_argument("bundle")
    publish.add_argument("--registry", required=True)
    publish.add_argument("--allow-untrusted", action="store_true")
    add_trust_options(publish)


def _add_memory_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("memory", help="Manage Living Memory patches")
    mem_sub = p.add_subparsers(dest="memory_command")
    review = mem_sub.add_parser("review", help="Review pending memory patches")
    review.add_argument("--patch-id", default=None)
    review.add_argument("--store-path", default=".synapsekit_memory_patches.jsonl")
    apply_cmd = mem_sub.add_parser("apply", help="Apply a pending memory patch")
    apply_cmd.add_argument("patch_id")
    apply_cmd.add_argument("--store-path", default=".synapsekit_memory_patches.jsonl")
    revert_cmd = mem_sub.add_parser("revert", help="Revert an applied memory patch")
    revert_cmd.add_argument("patch_id")
    revert_cmd.add_argument("--store-path", default=".synapsekit_memory_patches.jsonl")
    log_cmd = mem_sub.add_parser("log", help="View memory patch history")
    log_cmd.add_argument("--status", default=None)
    log_cmd.add_argument("--limit", type=int, default=20)
    log_cmd.add_argument(
        "--format", dest="output_format", choices=["table", "json"], default="table"
    )
    log_cmd.add_argument("--store-path", default=".synapsekit_memory_patches.jsonl")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""

    from .._loop import install_fast_loop

    install_fast_loop()
    parser = argparse.ArgumentParser(prog="synapsekit", description="SynapseKit CLI")
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
    _add_dream_parser(subparsers)
    _add_ambient_parser(subparsers)
    _add_sandbox_parser(subparsers)
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

    runners = {
        "serve": ("serve", "run_serve"),
        "test": ("test", "run_test"),
        "eval": ("eval", "run_eval"),
        "finetune": ("finetune", "run_finetune"),
        "graph-builder": ("graph_builder", "run_graph_builder"),
        "benchmark": ("benchmark", "run_benchmark"),
        "bench": ("bench", "run_bench"),
        "edge": ("edge", "run_edge"),
        "mesh": ("mesh", "run_mesh"),
        "shell": ("shell", "run_shell"),
        "dream": ("dream", "run_dream"),
        "ambient": ("ambient", "run_ambient"),
        "sandbox": ("sandbox", "run_sandbox"),
        "agent": ("agent", "run_agent"),
        "hive": ("hive", "run_hive"),
        "memory": ("memory", "run_memory"),
        "ui": ("ui", "run_ui"),
        "plugin": ("plugins", "run_plugin"),
        "audit": ("audit", "run_audit"),
    }
    if args.command not in runners:
        parser.print_help()
        raise SystemExit(1)
    module_name, function_name = runners[args.command]
    module = __import__(f"{__package__}.{module_name}", fromlist=[function_name])
    getattr(module, function_name)(args)


if __name__ == "__main__":
    main()
