"""``synapsekit sandbox`` lifecycle commands."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ..sandbox import DiffBundle, PCSandbox


def build_sandbox_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("sandbox", help="Manage isolated PC Twin sessions")
    commands = parser.add_subparsers(dest="sandbox_command")

    spawn = commands.add_parser("spawn", help="Create and start a sandbox session")
    spawn.add_argument("--base", default="current_user", help="Host root to snapshot")
    spawn.add_argument(
        "--backend", default="docker", choices=["docker", "orbstack", "lima", "firecracker", "fake"]
    )
    spawn.add_argument("--network", default="none", choices=["none", "egress_only"])
    spawn.add_argument(
        "--include", action="append", default=[], help="Relative include path (repeatable)"
    )
    spawn.add_argument(
        "--exclude", action="append", default=None, help="Relative exclude path (repeatable)"
    )
    spawn.add_argument("--state-dir", default=None, help="Session metadata directory")
    spawn.add_argument("--image", default="python:3.12-slim", help="Docker image")
    spawn.add_argument("--format", dest="output_format", choices=["text", "json"], default="text")

    diff = commands.add_parser("diff", help="Generate a diff bundle for a session")
    diff.add_argument("session_id")
    diff.add_argument("--state-dir", default=None)
    diff.add_argument("--output", default=None, help="Write a .diff.zip bundle")
    diff.add_argument("--format", dest="output_format", choices=["text", "json"], default="text")

    apply = commands.add_parser("apply", help="Apply an evaluated diff bundle")
    apply.add_argument("bundle")
    apply.add_argument("--receipt", required=True, help="JSON evaluation receipt")
    apply.add_argument("--yes", action="store_true", help="Confirm host mutation")

    discard = commands.add_parser("discard", help="Discard a sandbox session")
    discard.add_argument("session_id")
    discard.add_argument("--state-dir", default=None)


def run_sandbox(args: argparse.Namespace) -> None:
    command = getattr(args, "sandbox_command", None)
    if command == "spawn":
        _run_spawn(args)
        return
    if command == "diff":
        _run_diff(args)
        return
    if command == "apply":
        _run_apply(args)
        return
    if command == "discard":
        _run_discard(args)
        return
    raise SystemExit("Missing sandbox subcommand. Use: spawn, diff, apply, or discard")


def _run_spawn(args: argparse.Namespace) -> None:
    sandbox = PCSandbox(
        base=args.base,
        backend=args.backend,
        network=args.network,
        include=tuple(args.include),
        exclude=None if args.exclude is None else tuple(args.exclude),
        state_dir=args.state_dir,
        image=args.image,
    )
    environment = asyncio.run(sandbox.start())
    payload = {
        "session_id": environment.session_id,
        "backend": environment.handle.backend,
        "state": environment.state.value,
        "host_root": str(environment.host_root),
        "work_root": str(environment.work_root),
        "base_fingerprint": environment.baseline.fingerprint,
    }
    if args.output_format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(f"Sandbox: {payload['session_id']}")
        print(f"Backend: {payload['backend']}")
        print(f"Workspace: {payload['work_root']}")


def _run_diff(args: argparse.Namespace) -> None:
    sandbox = asyncio.run(PCSandbox.attach(args.session_id, state_dir=args.state_dir))
    environment = sandbox._environment
    if environment is None:
        raise SystemExit("Sandbox session is not available.")
    diff = asyncio.run(environment.diff_against_host())
    if args.output:
        DiffBundle.write(diff, args.output)
    preview = diff.preview()
    payload = {
        "session_id": diff.sandbox_id,
        "digest": diff.digest,
        "bundle": str(Path(args.output).resolve()) if args.output else None,
        "preview": {
            "changes": preview.changes,
            "additions": preview.additions,
            "modifications": preview.modifications,
            "deletions": preview.deletions,
            "directories": preview.directories,
            "total_bytes": preview.total_bytes,
        },
    }
    if args.output_format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(f"Diff: {payload['digest']}")
        print(f"Changes: {preview.changes}")
        if args.output:
            print(f"Bundle: {payload['bundle']}")


def _run_apply(args: argparse.Namespace) -> None:
    if not args.yes:
        raise SystemExit("Refusing host mutation without --yes.")
    bundle = DiffBundle.read(args.bundle)
    receipt_data = json.loads(Path(args.receipt).read_text(encoding="utf-8"))
    receipt = SimpleNamespace(
        passed=bool(receipt_data.get("passed")),
        diff_sha256=receipt_data.get("diff_sha256"),
    )
    asyncio.run(bundle.apply(receipt))
    print(f"Applied diff {bundle.digest}")


def _run_discard(args: argparse.Namespace) -> None:
    sandbox = asyncio.run(PCSandbox.attach(args.session_id, state_dir=args.state_dir))
    asyncio.run(sandbox.discard())
    print(f"Discarded sandbox {args.session_id}")
