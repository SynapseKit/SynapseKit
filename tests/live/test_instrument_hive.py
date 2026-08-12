"""Auto-instrumentation of Hive Mode (pooled memory) → SynapseKit Live.

Real objects only — a real in-process aggregator + SQLite store and a real
HiveClient. No server, no mocks: instrument the classes and toggle the bus.
"""

from __future__ import annotations

import inspect
import random
from pathlib import Path

import pytest

from synapsekit.hive import (
    DifferentialPrivacy,
    HiveAggregator,
    HiveClient,
    InProcessHiveTransport,
    PrivacyConfig,
    ShareScope,
    SQLiteHiveStore,
)
from synapsekit.live import bus
from synapsekit.live.instrument import instrument_all


@pytest.fixture(autouse=True)
def _instrumented_bus():
    instrument_all()
    was = bus.enabled
    bus.enabled = True
    bus.clear()
    yield
    bus.enabled = was


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
cover the implementation.
""",
        encoding="utf-8",
    )
    return path


def _client(aggregator: HiveAggregator, tmp_path: Path) -> HiveClient:
    return HiveClient(
        scope=ShareScope.TEAM,
        team_id="synapsekit",
        contributor_id="alice",
        cache_path=tmp_path / "alice.json",
        privacy=PrivacyConfig(epsilon=1.0, budget_limit=4.0, minimum_cohort=3),
        transport=InProcessHiveTransport(aggregator),
        dp=DifferentialPrivacy(rng=random.Random("alice")),
    )


def test_contribute_stays_a_coroutine_after_instrumentation() -> None:
    assert inspect.iscoroutinefunction(HiveClient.contribute)
    assert inspect.iscoroutinefunction(HiveClient.suggestions_for)
    assert inspect.iscoroutinefunction(HiveClient.withdraw)


async def test_hive_lifecycle_publishes_events(tmp_path: Path) -> None:
    source = _memory_file(tmp_path / "memory.md")
    client = _client(HiveAggregator(SQLiteHiveStore(":memory:")), tmp_path)

    await client.contribute([source])
    await client.suggestions_for()
    await client.withdraw()

    kinds = [e["kind"] for e in bus.history()]
    assert "hive.contribute" in kinds
    assert "hive.suggestions" in kinds
    assert "hive.withdraw" in kinds
    contribute = [e for e in bus.history() if e["kind"] == "hive.contribute"][-1]
    assert contribute["attributes"]["scope_id"] == "team:synapsekit"
    assert contribute["status"] == "ok"
    assert contribute["duration_ms"] >= 0
