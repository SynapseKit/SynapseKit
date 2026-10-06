#!/usr/bin/env python3
"""One-shot backfill of currently-open GitHub issues + PRs to per-label Discord channels.

Reads all open issues/PRs from `gh`, then for each label that has a configured
webhook (env var `DISCORD_WEBHOOK_<UPPER>`), posts an embed to that channel.
Idempotent via a JSON state file: each (issue_number, label) pair is posted at
most once unless `--force` is given.

Usage:
    # Set webhook URLs (any subset — missing ones are skipped):
    export DISCORD_WEBHOOK_V20=https://discord.com/api/webhooks/...
    export DISCORD_WEBHOOK_RESEARCH=https://discord.com/api/webhooks/...

    # Dry-run first
    python scripts/backfill_discord.py --dry-run

    # Then for real
    python scripts/backfill_discord.py

    # Re-post specific labels
    python scripts/backfill_discord.py --labels v2.0,research --force
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

REPO = os.environ.get("BACKFILL_REPO", "SynapseKit/SynapseKit")
STATE_FILE = Path(os.environ.get("BACKFILL_STATE", ".discord_backfill_state.json"))
DEFAULT_SLEEP = 1.2  # seconds between posts (Discord rate-limit headroom)

LABEL_CONFIG: dict[str, dict[str, Any]] = {
    "v1.8":          {"channel": "v1-8-current",         "emoji": "🛠",  "color": 3447003},
    "v2.0":          {"channel": "v2-0-futuristic-leap", "emoji": "🚀",  "color": 10038562},
    "v2.1":          {"channel": "v2-1-knowledge-mesh",  "emoji": "🧠",  "color": 5793266},
    "post-v2.0":     {"channel": "post-v2-0",            "emoji": "📚",  "color": 15105570},
    "research":      {"channel": "research",             "emoji": "🔬",  "color": 10181046},
    "benchmark":     {"channel": "benchmark",            "emoji": "📊",  "color": 3066993},
    "agents":        {"channel": "agents",               "emoji": "🤖",  "color": 15844367},
    "retrieval":     {"channel": "retrieval",            "emoji": "🔍",  "color": 1752220},
    "observability": {"channel": "observability",        "emoji": "📈",  "color": 2123412},
    "security":      {"channel": "security",             "emoji": "🔒",  "color": 15158332},
    "performance":   {"channel": "performance",          "emoji": "⚡",  "color": 16776960},
    "ui":            {"channel": "ui",                   "emoji": "🎨",  "color": 15418782},
    "book":          {"channel": "book",                 "emoji": "📖",  "color": 8421504},
    "paper":         {"channel": "paper",                "emoji": "📝",  "color": 9807270},
    "documentation": {"channel": "documentation",        "emoji": "📘",  "color": 3447003},
    "ambient":       {"channel": "ambient",              "emoji": "🌌",  "color": 7506394},
    "protocol":      {"channel": "protocol",             "emoji": "🔗",  "color": 1942002},
    "community":     {"channel": "community",            "emoji": "👥",  "color": 5763719},
}


def env_var(label: str) -> str:
    sanitized = label.upper().replace(".", "").replace("-", "_").replace(" ", "_")
    return f"DISCORD_WEBHOOK_{sanitized}"


def load_state() -> set[str]:
    if not STATE_FILE.exists():
        return set()
    return set(json.loads(STATE_FILE.read_text()))


def save_state(state: set[str]) -> None:
    STATE_FILE.write_text(json.dumps(sorted(state), indent=2))


def fetch_open_items() -> list[dict[str, Any]]:
    out = subprocess.check_output(
        [
            "gh", "issue", "list", "--repo", REPO,
            "--state", "open", "--limit", "500",
            "--json", "number,title,labels,url,author",
        ]
    )
    issues = json.loads(out)
    out = subprocess.check_output(
        [
            "gh", "pr", "list", "--repo", REPO,
            "--state", "open", "--limit", "500",
            "--json", "number,title,labels,url,author",
        ]
    )
    prs = json.loads(out)
    for item in issues:
        item["kind"] = "Issue"
    for item in prs:
        item["kind"] = "PR"
    return issues + prs


def post(webhook_url: str, payload: dict[str, Any]) -> tuple[int, str]:
    req = urllib.request.Request(
        webhook_url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "SynapseKit-Backfill (https://github.com/SynapseKit/SynapseKit, 1.0)",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def build_embed(item: dict[str, Any], label: str, config: dict[str, Any]) -> dict[str, Any]:
    author = (item.get("author") or {}).get("login") or "anonymous"
    return {
        "embeds": [
            {
                "title": f"{config['emoji']} {item['kind']} #{item['number']} — {label}",
                "description": f"**{item['title']}**\n\n[View on GitHub]({item['url']})",
                "color": config["color"],
                "footer": {"text": f"by {author}"},
            }
        ]
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Ignore state file")
    parser.add_argument("--labels", help="Comma-separated subset (default: all configured)")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP)
    args = parser.parse_args()

    selected = set(args.labels.split(",")) if args.labels else set(LABEL_CONFIG)
    state = set() if args.force else load_state()

    webhooks: dict[str, str] = {}
    for label in selected:
        if label not in LABEL_CONFIG:
            print(f"⚠ unknown label '{label}' — skipping", file=sys.stderr)
            continue
        url = os.environ.get(env_var(label))
        if url:
            webhooks[label] = url
        else:
            print(f"⊘ no {env_var(label)} set — skipping label '{label}'")

    if not webhooks:
        print("No webhooks configured. Set DISCORD_WEBHOOK_<LABEL> env vars and rerun.")
        return 1

    items = fetch_open_items()
    print(f"Found {len(items)} open items. Routing to {len(webhooks)} configured channels.")

    posted = 0
    skipped = 0
    failed = 0
    for item in items:
        labels_on_item = [label["name"] for label in item.get("labels", [])]
        for label in labels_on_item:
            if label not in webhooks:
                continue
            key = f"{item['number']}::{label}"
            if key in state and not args.force:
                skipped += 1
                continue
            payload = build_embed(item, label, LABEL_CONFIG[label])
            if args.dry_run:
                print(f"  DRY {item['kind']} #{item['number']} → #{LABEL_CONFIG[label]['channel']}")
                posted += 1
                continue
            status, body = post(webhooks[label], payload)
            if 200 <= status < 300:
                print(f"  ✓ {item['kind']} #{item['number']} → #{LABEL_CONFIG[label]['channel']}")
                state.add(key)
                posted += 1
            else:
                print(f"  ✗ {item['kind']} #{item['number']} → '{label}' HTTP {status}: {body[:200]}")
                failed += 1
            time.sleep(args.sleep)

    if not args.dry_run:
        save_state(state)

    print(f"\nDone. posted={posted} skipped={skipped} failed={failed}")
    print(f"State persisted to {STATE_FILE}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
