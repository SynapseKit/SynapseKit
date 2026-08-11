"""CLI entry points for the Agent OS Shell."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
from pathlib import Path
from typing import Any

from synapsekit.shell import (
    CachedPlanner,
    CompletionEngine,
    JsonAmbientContext,
    NullAmbientContext,
    RuleBasedPlanner,
    ShellHistory,
    ShellSession,
    TranslationCache,
    generate_signing_key,
    get_adapter,
    load_signing_policy,
    result_to_dict,
)


def _add_shell_options(parser: argparse.ArgumentParser, *, suppress_defaults: bool = False) -> None:
    default = argparse.SUPPRESS if suppress_defaults else None
    parser.add_argument("--cwd", default=default, help="Working directory for commands")
    parser.add_argument("--shell", default=default or "auto", help="bash, zsh, fish, or powershell")
    parser.add_argument("--ambient-status", default=default, help="Ambient Agent status JSON path")
    parser.add_argument("--history-path", default=default, help="SQLite shell history path")
    parser.add_argument(
        "--translation-cache", default=default, help="SQLite translation cache path"
    )
    parser.add_argument(
        "--mesh-root", default=default, help="Project root to query with KnowledgeMesh"
    )
    parser.add_argument(
        "--no-mesh",
        action="store_true",
        default=default or False,
        help="Disable KnowledgeMesh context",
    )
    parser.add_argument(
        "--planner-module", default=default, help="Planner import path, e.g. package.module:planner"
    )
    parser.add_argument(
        "--signing-key", default=default, help="Raw Ed25519 private key for destructive approvals"
    )
    parser.add_argument("--key-id", default=default, help="Stable audit signing key id")
    parser.add_argument("--audit-dir", default=default, help="Directory for signed audit bundles")
    parser.add_argument(
        "--timeout", type=float, default=default or 30.0, help="Command timeout in seconds"
    )
    parser.add_argument(
        "--max-output", type=int, default=default or 1_000_000, help="Output cap per command"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        default=default or False,
        help="Approve destructive commands after signed preflight",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=default or False,
        help="Plan and preview without executing",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=default or False,
        dest="json_output",
        help="Emit machine-readable JSON",
    )


def build_shell_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    parser = subparsers.add_parser(
        "shell",
        help="Use a context-aware hybrid natural-language and shell session",
    )
    _add_shell_options(parser)
    shell_sub = parser.add_subparsers(dest="shell_command")

    run = shell_sub.add_parser("run", help="Plan and execute one mixed input line")
    run.add_argument("text", nargs="+", help="Shell text and/or quoted natural language")
    _add_shell_options(run, suppress_defaults=True)

    init = shell_sub.add_parser("init", help="Print the integration script for a shell")
    init.add_argument("shell", choices=["bash", "zsh", "fish", "powershell"])

    complete = shell_sub.add_parser("complete", help="Return local and mesh-enriched completions")
    complete.add_argument("prefix", nargs="*", default=[])
    complete.add_argument("--cwd", default=None)
    complete.add_argument("--mesh-root", default=None)
    complete.add_argument("--no-mesh", action="store_true")

    history = shell_sub.add_parser("history", help="Search local redacted shell history")
    history_sub = history.add_subparsers(dest="history_command")
    search = history_sub.add_parser("search", help="Search history by terms")
    search.add_argument("query", nargs="*", default=[])
    search.add_argument("--path", default=None)
    search.add_argument("--limit", type=int, default=20)
    recent = history_sub.add_parser("recent", help="Show recent history")
    recent.add_argument("--path", default=None)
    recent.add_argument("--limit", type=int, default=20)

    keygen = shell_sub.add_parser("keygen", help="Generate a local Ed25519 shell audit key")
    keygen.add_argument("private_key")
    keygen.add_argument("--public-key", default=None)

    status = shell_sub.add_parser("status", help="Show shell integration and context status")
    status.add_argument("--cwd", default=None)
    status.add_argument("--shell", default="auto")
    status.add_argument("--mesh-root", default=None)
    status.add_argument("--no-mesh", action="store_true")


def _load_planner(module_path: str | None) -> Any:
    if not module_path:
        return RuleBasedPlanner()
    module_name, separator, attribute = module_path.partition(":")
    if not separator or not attribute:
        raise ValueError("--planner-module must be MODULE:ATTRIBUTE")
    value = getattr(importlib.import_module(module_name), attribute)
    return value() if callable(value) and not hasattr(value, "plan") else value


def _make_mesh(args: Any) -> Any | None:
    if getattr(args, "no_mesh", False):
        return None
    try:
        from synapsekit.mesh import KnowledgeMesh, MeshConfig

        root = Path(args.mesh_root or args.cwd or Path.cwd()).expanduser()
        return KnowledgeMesh(MeshConfig(roots=[root]))
    except (ImportError, OSError, ValueError):
        return None


def _make_session(args: Any) -> ShellSession:
    cwd = Path(args.cwd or Path.cwd()).expanduser().resolve()
    shell = get_adapter(args.shell).kind.value
    history = ShellHistory(
        args.history_path or Path.home() / ".synapsekit" / "shell" / "history.sqlite3"
    )
    planner = CachedPlanner(
        _load_planner(args.planner_module),
        TranslationCache(
            args.translation_cache or Path.home() / ".synapsekit" / "shell" / "translations.sqlite3"
        ),
        model=args.planner_module or "rules",
    )
    signing_policy = (
        load_signing_policy(args.signing_key, key_id=args.key_id) if args.signing_key else None
    )
    ambient = (
        JsonAmbientContext(args.ambient_status) if args.ambient_status else NullAmbientContext()
    )
    return ShellSession(
        planner=planner,
        cwd=cwd,
        shell=shell,
        mesh=_make_mesh(args),
        ambient=ambient,
        history=history,
        signing_policy=signing_policy,
        audit_dir=args.audit_dir,
        timeout=args.timeout,
        max_output_bytes=args.max_output,
    )


async def _confirm(step: Any, preview: str) -> bool:
    print("\nDestructive command preview:")
    print(preview)
    answer = await asyncio.to_thread(input, "Execute this command? [y/N] ")
    return answer.strip().casefold() in {"y", "yes"}


async def _run_once(session: ShellSession, args: Any, text: str) -> int:
    result = await session.run(text, confirm=_confirm, assume_yes=args.yes, dry_run=args.dry_run)
    if args.json_output:
        print(json.dumps(result_to_dict(result), indent=2, default=str))
    else:
        if result.plan.steps:
            print(f"Plan: {result.plan.summary}")
            for warning in result.plan.warnings:
                print(f"Preview: {warning}")
        for item in result.commands:
            if item.stdout:
                print(item.stdout, end="" if item.stdout.endswith("\n") else "\n")
            if item.stderr:
                print(item.stderr, end="" if item.stderr.endswith("\n") else "\n")
        if result.audit_path:
            print(f"Audit: {result.audit_path}")
        if result.error:
            print(f"Error: {result.error}")
    return 0 if result.ok else 1


async def _repl(session: ShellSession, args: Any) -> int:
    prompt = get_adapter(args.shell).prompt()
    while True:
        try:
            line = await asyncio.to_thread(input, prompt)
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line.strip():
            continue
        if line.strip().casefold() in {"exit", "quit"}:
            return 0
        code = await _run_once(session, args, line)
        if code:
            print("The command was not completed.")


def run_shell(args: Any) -> None:
    command = args.shell_command
    if command == "init":
        print(get_adapter(args.shell).init_script(), end="")
        return
    if command == "complete":
        engine = CompletionEngine(_make_mesh(args))
        prefix = " ".join(args.prefix)
        values = asyncio.run(engine.complete(prefix, cwd=args.cwd or Path.cwd()))
        print("\n".join(values))
        return
    if command == "keygen":
        print(json.dumps(generate_signing_key(args.private_key, args.public_key), indent=2))
        return
    if command == "history":
        history = ShellHistory(getattr(args, "path", None))
        query = " ".join(getattr(args, "query", []))
        rows = asyncio.run(
            history.search(query, limit=args.limit)
            if args.history_command == "search"
            else history.recent(limit=args.limit)
        )
        print(json.dumps(rows, indent=2, default=str))
        return
    if command == "status":
        payload: dict[str, Any] = {
            "shell": get_adapter(args.shell).name,
            "cwd": str(Path(args.cwd or Path.cwd()).resolve()),
        }
        mesh = _make_mesh(args)
        if mesh is not None:
            payload["mesh"] = mesh.status().__dict__
        print(json.dumps(payload, indent=2, default=str))
        return
    session = _make_session(args)
    text = " ".join(args.text) if command == "run" else None
    code = (
        asyncio.run(_run_once(session, args, text))
        if text is not None
        else asyncio.run(_repl(session, args))
    )
    if code:
        raise SystemExit(code)


def main(argv: list[str] | None = None) -> None:
    """Standalone ``synshell`` entry point."""
    parser = argparse.ArgumentParser(prog="synshell")
    _add_shell_options(parser)
    parser.add_argument("text", nargs="*")
    args = parser.parse_args(argv)
    session = _make_session(args)
    text = " ".join(args.text)
    code = (
        asyncio.run(_run_once(session, args, text)) if text else asyncio.run(_repl(session, args))
    )
    if code:
        raise SystemExit(code)
