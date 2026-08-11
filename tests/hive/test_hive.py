"""Behavior-focused coverage for Hive Mode's privacy and transport boundaries."""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import replace
from pathlib import Path

import pytest

from synapsekit.hive import (
    DifferentialPrivacy,
    HiveAggregator,
    HiveAggregatorError,
    HiveClient,
    InProcessHiveTransport,
    PrivacyConfig,
    ShareScope,
    SQLiteHiveStore,
)


def _memory_file(path: Path) -> Path:
    path.write_text(
        """---
ump_version: '1.0'
name: local-memory
type: project
scope: project
visibility: local
---

FastAPI services use exponential backoff and idempotency. pytest and ruff
cover the implementation. This text must never be stored by Hive.
""",
        encoding="utf-8",
    )
    return path


def _client(
    root: Path,
    aggregator: HiveAggregator,
    tmp_path: Path,
    contributor: str,
    *,
    encryption_key: bytes | None = None,
) -> HiveClient:
    return HiveClient(
        scope=ShareScope.TEAM,
        team_id="synapsekit",
        contributor_id=contributor,
        cache_path=tmp_path / f"{contributor}.json",
        privacy=PrivacyConfig(epsilon=1.0, budget_limit=4.0, minimum_cohort=3),
        transport=InProcessHiveTransport(aggregator),
        encryption_key=encryption_key,
        dp=DifferentialPrivacy(rng=random.Random(contributor)),
    )


def test_pseudonymized_contributors_and_scope_ids_are_stable(tmp_path: Path) -> None:
    first = HiveClient(
        scope="team",
        team_id="Acme Team",
        contributor_id="alice",
        cache_path=tmp_path / "first.json",
    )
    second = HiveClient(
        scope="team",
        team_id="Acme Team",
        contributor_id="alice",
        cache_path=tmp_path / "second.json",
        pseudonymizer=first.pseudonymizer,
    )
    assert first.scope_id == "team:acme-team"
    assert first.contributor_id == second.contributor_id
    assert first.contributor_id != "alice"


def test_three_contributors_produce_aggregate_suggestions_without_raw_text(tmp_path: Path) -> None:
    source = _memory_file(tmp_path / "memory.md")
    store = SQLiteHiveStore(":memory:")
    aggregator = HiveAggregator(store)

    for contributor in ("alice", "bob", "carol"):
        client = _client(source, aggregator, tmp_path, contributor)
        asyncio.run(client.contribute([source]))

    suggestions = aggregator.suggestions(scope_id="team:synapsekit", minimum_cohort=3)
    assert suggestions
    assert any(item.key == "framework:fastapi" for item in suggestions)
    stored = store.list(scope_id="team:synapsekit")
    assert all(stored_item.payload is not None for stored_item in stored)
    assert all(
        "This text must never" not in json.dumps(stored_item.to_dict()) for stored_item in stored
    )


def test_encrypted_contribution_is_decrypted_only_by_configured_aggregator(tmp_path: Path) -> None:
    source = _memory_file(tmp_path / "memory.md")
    key = b"0123456789abcdef0123456789abcdef"
    store = SQLiteHiveStore(":memory:")
    aggregator = HiveAggregator(store, encryption_key=key)
    client = _client(source, aggregator, tmp_path, "alice", encryption_key=key)
    asyncio.run(client.contribute([source]))
    assert store.list(scope_id="team:synapsekit")[0].payload is not None


def test_tampered_signed_contribution_is_rejected(tmp_path: Path) -> None:
    source = _memory_file(tmp_path / "memory.md")
    aggregator = HiveAggregator(SQLiteHiveStore(":memory:"))
    client = _client(source, aggregator, tmp_path, "alice")
    mined = client.miner.mine(
        [source], include=client.include, privacy_filter=client.privacy_filter
    )
    payload = replace(
        client._sign(
            __import__("synapsekit.hive").hive.ContributionPayload(
                scope=client.scope,
                scope_id=client.scope_id,
                patterns=mined.patterns,
                epsilon=1.0,
                delta=1e-6,
            )
        ),
        signature="invalid",
    )
    with pytest.raises(HiveAggregatorError):
        aggregator.submit(payload)


def test_offline_suggestions_use_last_known_cache(tmp_path: Path) -> None:
    cache = tmp_path / "cache.json"
    cache.write_text(
        json.dumps(
            {
                "suggestions": [
                    {
                        "key": "practice:idempotency",
                        "category": "practice",
                        "statement": "Use idempotency.",
                        "prevalence": 0.8,
                        "confidence": 0.7,
                        "contributor_count": 4,
                        "cohort_size": 5,
                        "scope": "team",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    client = HiveClient(scope="team", team_id="synapsekit", cache_path=cache)
    suggestions = asyncio.run(client.suggestions_for())
    assert suggestions[0].key == "practice:idempotency"
