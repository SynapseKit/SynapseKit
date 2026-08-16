"""CLI entry points for explicit Dream Mode runs."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from ..dream import DreamConfig, DreamMode, DreamStateStore, render_briefing
from ..mesh import KnowledgeMesh, MeshConfig


def build_dream_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    parser = subparsers.add_parser("dream", help="Run local-first overnight self-reflection")
    dream_sub = parser.add_subparsers(dest="dream_command")

    run = dream_sub.add_parser("run", help="Run one bounded Dream Mode cycle")
    run.add_argument("--state-path", default="~/.synapsekit/dream/state.sqlite3")
    run.add_argument("--audit-dir", default="~/.synapsekit/dream/audit")
    run.add_argument(
        "--trace-bundle", action="append", default=[], help="Local signed audit bundle"
    )
    run.add_argument("--memory-path", action="append", default=[], help="Memory markdown file")
    run.add_argument("--mesh-root", action="append", default=[], help="Local KnowledgeMesh root")
    run.add_argument("--schedule", default="idle_30m or 02:00")
    run.add_argument("--budget-tokens", type=int, default=100_000)
    run.add_argument("--stale-after-days", type=int, default=90)
    run.add_argument(
        "--force", action="store_true", help="Bypass schedule; power policy remains active"
    )
    run.add_argument("--json", action="store_true", dest="json_output")

    status = dream_sub.add_parser("status", help="Show the latest local Dream Mode briefing")
    status.add_argument("--state-path", default="~/.synapsekit/dream/state.sqlite3")
    status.add_argument("--json", action="store_true", dest="json_output")


def run_dream(args: Any) -> None:
    state_path = Path(args.state_path).expanduser()
    if args.dream_command == "status":
        # Read-only: use the store directly so we never generate a signing
        # key or otherwise touch the write path just to print a briefing.
        store = DreamStateStore(state_path)
        try:
            result = store.last_run()
            print(
                json.dumps(result.to_dict() if result else None, indent=2)
                if args.json_output
                else render_briefing(result)
            )
        finally:
            store.close()
        return

    if args.dream_command != "run":
        raise SystemExit("choose a Dream Mode command: run or status")
    mesh = None
    if args.mesh_root:
        mesh = KnowledgeMesh(MeshConfig(roots=[Path(root).expanduser() for root in args.mesh_root]))
    mode = DreamMode(
        config=DreamConfig(
            schedule=args.schedule,
            budget_tokens=args.budget_tokens,
            stale_after_days=args.stale_after_days,
            state_path=state_path,
            audit_dir=args.audit_dir,
        ),
        memory_paths=args.memory_path,
        mesh=mesh,
    )
    try:
        result = asyncio.run(mode.run_once(force=args.force, trace_bundles=args.trace_bundle))
        print(
            json.dumps(result.to_dict(), indent=2)
            if args.json_output
            else mode.morning_briefing(result)
        )
    finally:
        mode.close()
