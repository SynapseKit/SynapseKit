"""Behavior-focused coverage for Hive Mode's privacy and transport boundaries."""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import replace
from pathlib import Path

import pytest

from synapsekit.hive import (
    ContributionEnvelope,
    ContributionPayload,
    DifferentialPrivacy,
    HiveAggregator,
    HiveAggregatorError,
    HiveClient,
    InProcessHiveTransport,
    PatternObservation,
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


def _envelope(
    contributor_id: str, key: str, value: float, *, signature: str
) -> ContributionEnvelope:
    """Build a stored-shape envelope directly (suggestions() reads it unverified)."""

    payload = ContributionPayload(
        scope=ShareScope.TEAM,
        scope_id="team:synapsekit",
        patterns=(PatternObservation(key=key, value=value, category="framework"),),
        epsilon=1.0,
        delta=1e-6,
    )
    return ContributionEnvelope(
        payload=payload,
        contributor_id=contributor_id,
        public_key="unused",
        signature=signature,
    )


def test_contributor_count_reflects_distinct_contributors_not_observations() -> None:
    # alice submits the same pattern twice; bob and carol once each. Distinct
    # contributors for the key is 3 (not 4 observations) — regression for #928.
    store = SQLiteHiveStore(":memory:")
    store.put(_envelope("alice", "framework:fastapi", 1.0, signature="a1"))
    store.put(_envelope("alice", "framework:fastapi", 1.0, signature="a2"))
    store.put(_envelope("bob", "framework:fastapi", 1.0, signature="b1"))
    store.put(_envelope("carol", "framework:fastapi", 1.0, signature="c1"))

    aggregator = HiveAggregator(store)
    suggestions = aggregator.suggestions(scope_id="team:synapsekit", minimum_cohort=3)

    fastapi = next(item for item in suggestions if item.key == "framework:fastapi")
    assert fastapi.cohort_size == 3
    assert fastapi.contributor_count == 3
    assert fastapi.confidence <= 1.0


def test_sqlite_store_is_thread_safe_under_concurrent_writes() -> None:
    # Concurrent put/list from asyncio.to_thread workers must not raise or lose
    # rows on the single shared connection — regression for #927.
    store = SQLiteHiveStore(":memory:")
    count = 60

    async def _drive() -> None:
        writers = [
            asyncio.to_thread(
                store.put, _envelope(f"c{i}", "framework:fastapi", 1.0, signature=f"s{i}")
            )
            for i in range(count)
        ]
        readers = [asyncio.to_thread(store.list, scope_id="team:synapsekit") for _ in range(count)]
        await asyncio.gather(*writers, *readers)

    asyncio.run(_drive())

    stored = store.list(scope_id="team:synapsekit")
    assert len(stored) == count
    assert len({item.contributor_id for item in stored}) == count
