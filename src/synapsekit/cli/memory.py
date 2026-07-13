"""CLI handler for ``synapsekit memory`` subcommands."""

from __future__ import annotations

import argparse
import json
import sys


def run_memory(args: argparse.Namespace) -> None:
    """Dispatch memory subcommands."""
    cmd = getattr(args, "memory_command", None)
    if cmd == "review":
        _run_review(args)
    elif cmd == "apply":
        _run_apply(args)
    elif cmd == "revert":
        _run_revert(args)
    elif cmd == "log":
        _run_log(args)
    else:
        print("Usage: synapsekit memory {review,apply,revert,log}", file=sys.stderr)
        sys.exit(1)


def _run_review(args: argparse.Namespace) -> None:
    from ..memory.patch_store import PatchStore

    store = PatchStore(args.store_path)
    pending = store.pending_patches()

    patch_id = getattr(args, "patch_id", None)
    if patch_id:
        # Review a specific patch
        patch = store.get(patch_id)
        if patch is None:
            print(f"Patch {patch_id!r} not found.", file=sys.stderr)
            sys.exit(1)
        _display_patch(patch)
        return

    if not pending:
        print("No pending memory patches to review.")
        return

    # List all pending patches
    print(f"{'ID':<18} {'File':<30} {'Category':<12} {'Created':<22} Rationale")
    print("-" * 100)
    for p in pending:
        rationale_preview = (
            p.rationale[:40] + "…" if len(p.rationale) > 40 else p.rationale
        )
        print(
            f"{p.patch_id:<18} {p.file_path:<30} {p.category:<12} "
            f"{p.created_at[:19]:<22} {rationale_preview}"
        )
    print(f"\n{len(pending)} pending patch(es). Use --patch-id <ID> to review details.")


def _run_apply(args: argparse.Namespace) -> None:
    from datetime import datetime, timezone

    from ..memory.diff_engine import FileDiffEngine
    from ..memory.patch_store import PatchStore

    store = PatchStore(args.store_path)
    patch = store.get(args.patch_id)
    if patch is None:
        print(f"Patch {args.patch_id!r} not found.", file=sys.stderr)
        sys.exit(1)

    if patch.status != "pending":
        print(
            f"Patch {args.patch_id} has status {patch.status!r}, cannot apply.",
            file=sys.stderr,
        )
        sys.exit(1)

    applicable, reason = FileDiffEngine.validate_patch_applicable(
        patch.file_path, patch.before_content
    )
    if not applicable:
        print(f"Cannot apply: {reason}", file=sys.stderr)
        sys.exit(1)

    FileDiffEngine.apply_patch(patch.file_path, patch.after_content)
    patch.status = "applied"
    patch.applied_at = datetime.now(timezone.utc).isoformat()
    patch.sign()
    store.update(patch)
    print(f"Applied patch {patch.patch_id} to {patch.file_path}")


def _run_revert(args: argparse.Namespace) -> None:
    from datetime import datetime, timezone

    from ..memory.diff_engine import FileDiffEngine
    from ..memory.patch_store import PatchStore

    store = PatchStore(args.store_path)
    patch = store.get(args.patch_id)
    if patch is None:
        print(f"Patch {args.patch_id!r} not found.", file=sys.stderr)
        sys.exit(1)

    if patch.status != "applied":
        print(
            f"Patch {args.patch_id} has status {patch.status!r}, cannot revert.",
            file=sys.stderr,
        )
        sys.exit(1)

    FileDiffEngine.revert_to_content(patch.file_path, patch.before_content)
    patch.status = "reverted"
    patch.reverted_at = datetime.now(timezone.utc).isoformat()
    patch.sign()
    store.update(patch)
    print(f"Reverted patch {patch.patch_id} — restored {patch.file_path}")


def _run_log(args: argparse.Namespace) -> None:
    from ..memory.patch_store import PatchStore

    store = PatchStore(args.store_path)
    status_filter = getattr(args, "status", None)
    limit = getattr(args, "limit", 20)
    fmt = getattr(args, "output_format", "table")

    patches = store.list_by_status(status_filter, limit=limit)
    if not patches:
        print("No patches found.")
        return

    if fmt == "json":
        print(json.dumps([p.to_dict() for p in patches], indent=2, default=str))
        return

    print(f"{'ID':<18} {'Status':<12} {'File':<30} {'Created':<22} Rationale")
    print("-" * 100)
    for p in patches:
        rationale_preview = (
            p.rationale[:35] + "…" if len(p.rationale) > 35 else p.rationale
        )
        print(
            f"{p.patch_id:<18} {p.status:<12} {p.file_path:<30} "
            f"{p.created_at[:19]:<22} {rationale_preview}"
        )


def _display_patch(patch) -> None:  # type: ignore[no-untyped-def]
    """Display a single patch in detail."""
    print(f"Patch ID:     {patch.patch_id}")
    print(f"File:         {patch.file_path}")
    print(f"Status:       {patch.status}")
    print(f"Category:     {patch.category}")
    print(f"Created:      {patch.created_at}")
    if patch.signature:
        print(f"Signature:    {patch.signature[:16]}…")
    else:
        print("Signature:    (none)")
    print(f"Rationale:    {patch.rationale}")
    if patch.evidence_refs:
        print(f"Evidence:     {'; '.join(patch.evidence_refs[:3])}")
    print()
    print("--- Unified Diff ---")
    print(patch.unified_diff)
