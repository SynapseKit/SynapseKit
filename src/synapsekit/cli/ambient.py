"""``synapsekit ambient`` commands for the ambient agent daemon."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from synapsekit.ambient import AmbientDaemon, AmbientDaemonConfig
from synapsekit.observability import AuditLog


def _ambient_config(args: Any) -> AmbientDaemonConfig:
    defaults = AmbientDaemonConfig()
    return AmbientDaemonConfig(
        repo_root=Path(getattr(args, "repo", None) or Path.cwd()),
        poll_interval=float(getattr(args, "poll_interval", defaults.poll_interval)),
        min_confidence=float(getattr(args, "min_confidence", defaults.min_confidence)),
    )


def _print_payload(payload: Any, *, output_json: bool = False) -> None:
    if output_json:
        print(json.dumps(payload, indent=2))
        return
    if isinstance(payload, dict):
        for key, value in payload.items():
            print(f"{key}: {value}")
        return
    print(payload)


def run_ambient(args: Any) -> None:
    """Dispatch ``synapsekit ambient`` subcommands."""

    command = getattr(args, "ambient_command", None)
    output_json = bool(getattr(args, "json", False))

    if command == "start":
        daemon = AmbientDaemon(config=_ambient_config(args))
        status = daemon.start_sync()
        _print_payload(asdict(status), output_json=output_json)
        return

    if command == "stop":
        status = AmbientDaemon(config=_ambient_config(args)).stop_sync()
        _print_payload(asdict(status), output_json=output_json)
        return

    if command == "status":
        status = AmbientDaemon(config=_ambient_config(args)).status()
        _print_payload(asdict(status), output_json=output_json)
        return

    if command == "log":
        config = _ambient_config(args)
        audit_log = AuditLog(backend="jsonl", path=str(config.audit_path))
        entries = audit_log.query(limit=int(getattr(args, "limit", 20)))
        payload = [asdict(entry) for entry in entries]
        _print_payload(payload, output_json=output_json)
        return

    raise SystemExit("Missing ambient action. Use start, stop, status, or log.")


def build_ambient_parser(subparsers: Any) -> None:
    """Register the ``ambient`` parser with the top-level CLI."""

    parser = subparsers.add_parser("ambient", help="Observe local activity and proactively notify")
    parser.add_argument("--repo", default=None, help="Repo root to observe (default: cwd)")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    ambient_sub = parser.add_subparsers(dest="ambient_command")

    start_cmd = ambient_sub.add_parser("start", help="Start the ambient daemon")
    start_cmd.add_argument("--poll-interval", type=float, default=2.0, help="Poll interval seconds")
    start_cmd.add_argument(
        "--min-confidence", type=float, default=0.6, help="Minimum confidence to notify"
    )

    ambient_sub.add_parser("stop", help="Stop the ambient daemon")
    ambient_sub.add_parser("status", help="Show ambient daemon status")

    log_cmd = ambient_sub.add_parser("log", help="Show recent interventions")
    log_cmd.add_argument("--limit", type=int, default=20, help="Maximum entries to show")
