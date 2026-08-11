"""CLI commands for local and self-hosted Hive deployments."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any


def build_hive_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    parser = subparsers.add_parser(
        "hive", help="Contribute and inspect privacy-preserving Hive patterns"
    )
    commands = parser.add_subparsers(dest="hive_command")

    contribute = commands.add_parser(
        "contribute", help="Mine, privatize, sign, and upload local patterns"
    )
    _add_common(contribute)
    contribute.add_argument("roots", nargs="*", default=["."], help="Markdown roots to inspect")

    suggestions = commands.add_parser(
        "suggestions", help="Read aggregate suggestions with offline fallback"
    )
    _add_common(suggestions)
    suggestions.add_argument("--query", default=None)
    suggestions.add_argument("--limit", type=int, default=20)

    status = commands.add_parser(
        "status", help="Show local Hive budget and contribution transparency"
    )
    _add_common(status)

    withdraw = commands.add_parser(
        "withdraw", help="Revoke this contributor's stored contributions"
    )
    _add_common(withdraw)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scope", choices=["local", "team", "community"], default="local")
    parser.add_argument("--team-id", default=None)
    parser.add_argument("--contributor-id", default=None)
    parser.add_argument("--cache", default=None, help="Local Hive cache path")
    parser.add_argument(
        "--aggregator-db",
        default=None,
        help="Local SQLite aggregator path (defaults beside --cache)",
    )
    parser.add_argument("--service-url", default=None, help="Self-hosted Hive service URL")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--budget-limit", type=float, default=10.0)
    parser.add_argument("--minimum-cohort", type=int, default=3)


def run_hive(args: Any) -> None:
    from ..hive import (
        HiveAggregator,
        HiveClient,
        HttpHiveTransport,
        InProcessHiveTransport,
        PrivacyConfig,
        SQLiteHiveStore,
    )

    cache_path = Path(args.cache) if args.cache else Path.home() / ".synapsekit" / "hive.json"
    local_store = None
    if args.service_url:
        transport = HttpHiveTransport(args.service_url, api_key=args.api_key)
    else:
        database_path = (
            Path(args.aggregator_db) if args.aggregator_db else cache_path.with_suffix(".sqlite3")
        )
        local_store = SQLiteHiveStore(database_path)
        transport = InProcessHiveTransport(HiveAggregator(local_store))

    client = HiveClient(
        scope=args.scope,
        team_id=args.team_id,
        contributor_id=args.contributor_id,
        cache_path=cache_path,
        privacy=PrivacyConfig(
            epsilon=args.epsilon,
            budget_limit=args.budget_limit,
            minimum_cohort=args.minimum_cohort,
        ),
        transport=transport,
    )
    try:
        command = getattr(args, "hive_command", None)
        if command == "contribute":
            result = asyncio.run(client.contribute(args.roots))
            print(json.dumps({"contribution_id": result, "scope": client.scope_id}, indent=2))
        elif command == "suggestions":
            result = asyncio.run(client.suggestions_for(args.query, limit=args.limit))
            print(json.dumps([item.to_dict() for item in result], indent=2))
        elif command == "withdraw":
            result = asyncio.run(client.withdraw())
            print(json.dumps({"revoked": result}, indent=2))
        elif command == "status":
            result = asyncio.run(client.transparency())
            print(json.dumps(result.to_dict(), indent=2))
        else:
            raise SystemExit(
                "Missing hive subcommand. Use contribute, suggestions, status, or withdraw."
            )
    finally:
        if local_store is not None:
            local_store.close()
