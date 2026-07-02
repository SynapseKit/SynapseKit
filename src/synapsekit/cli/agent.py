"""``synapsekit agent`` commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..agents.self_improving import AgentEvolutionAuditLog


def run_agent(args: Any) -> None:
    subcommand = getattr(args, "agent_command", None)
    if subcommand == "inspect-evolution":
        _run_inspect_evolution(args)
        return
    raise SystemExit("Missing agent subcommand. Use: inspect-evolution")


def _run_inspect_evolution(args: Any) -> None:
    audit_path = Path(args.audit_path)
    if not audit_path.exists():
        print(f"No evolution audit log found at {audit_path}")
        return

    log = AgentEvolutionAuditLog(audit_path)
    rows = [
        patch
        for patch in log.list(limit=args.limit)
        if args.agent_id in {patch.before.get("agent_id"), patch.after.get("agent_id")}
        or patch.metadata.get("agent_id") == args.agent_id
        or args.agent_id == "all"
    ]

    if args.output_format == "json":
        print(json.dumps([patch.to_dict() for patch in rows], indent=2, default=str))
        return

    if not rows:
        print(f"No evolution patches found for agent '{args.agent_id}'.")
        return

    print()
    print(f"{'Patch':<12} {'Status':<12} {'Type':<18} {'Score':<8} {'Rollout':<8} Description")
    print("-" * 92)
    for patch in rows:
        score = f"{patch.eval_score:.3f}" if patch.eval_score is not None else "N/A"
        print(
            f"{patch.patch_id[:12]:<12} "
            f"{patch.status:<12} "
            f"{patch.patch_type:<18} "
            f"{score:<8} "
            f"{patch.rollout_pct:<8.1f} "
            f"{patch.description}"
        )
    print()
