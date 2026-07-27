from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from synapsekit.audit.signer import Ed25519SigningProvider
from synapsekit.marketplace import (
    FileAgentRegistry,
    InvalidAgentBundleError,
    SignedAgentReview,
    UntrustedPublisherError,
    pack_agent,
)


def _pack(tmp_path: Path, name: str, score: float, marker: str = "v1"):
    source = tmp_path / f"source-{name}-{marker}"
    (source / "evals").mkdir(parents=True)
    (source / "README.md").write_text(f"# {name}\n", encoding="utf-8")
    (source / "evals" / "suite.json").write_text("{}\n", encoding="utf-8")
    (source / "prompt.md").write_text(marker, encoding="utf-8")
    provider = Ed25519SigningProvider(key_id=f"publisher-{name}")
    bundle = pack_agent(
        source,
        tmp_path / f"{name}-{marker}.agent",
        name=name,
        version="1.0.0",
        author="Publisher",
        signing_provider=provider,
        eval_score=score,
    )
    return bundle, provider


def test_registry_requires_pinned_publisher_by_default(tmp_path: Path) -> None:
    bundle, _ = _pack(tmp_path, "reviewer", 0.8)

    with pytest.raises(UntrustedPublisherError):
        FileAgentRegistry(tmp_path / "registry").publish(bundle)


def test_registry_rejects_unsafe_review_lookup_path(tmp_path: Path) -> None:
    registry = FileAgentRegistry(tmp_path / "registry")

    with pytest.raises(ValueError, match="Unsafe agent name"):
        registry.reviews("..", "1.0.0")


def test_registry_publishes_immutable_bundle_and_static_index(tmp_path: Path) -> None:
    bundle, provider = _pack(tmp_path, "reviewer", 0.8)
    registry = FileAgentRegistry(tmp_path / "registry")
    trusted_keys = {provider.key_id: provider.public_key_bytes()}

    entry = registry.publish(bundle, trusted_keys=trusted_keys)
    duplicate = registry.publish(bundle, trusted_keys=trusted_keys)

    assert duplicate == entry
    assert registry.get("reviewer", "1.0.0") == entry
    assert registry.bundle_path("reviewer", "1.0.0").read_bytes() == bundle.read_bytes()
    assert registry.index_path.is_file()


def test_registry_refuses_replacing_an_existing_version(tmp_path: Path) -> None:
    first, first_provider = _pack(tmp_path, "reviewer", 0.8, marker="v1")
    second, second_provider = _pack(tmp_path, "reviewer", 0.9, marker="v2")
    registry = FileAgentRegistry(tmp_path / "registry")
    registry.publish(first, trusted_keys={first_provider.key_id: first_provider.public_key_bytes()})

    with pytest.raises(FileExistsError):
        registry.publish(
            second,
            trusted_keys={second_provider.key_id: second_provider.public_key_bytes()},
        )


def test_signed_reviews_feed_deterministic_eval_ranking(tmp_path: Path) -> None:
    first, first_provider = _pack(tmp_path, "first-agent", 0.9)
    second, second_provider = _pack(tmp_path, "second-agent", 0.6)
    registry = FileAgentRegistry(tmp_path / "registry")
    registry.publish(first, trusted_keys={first_provider.key_id: first_provider.public_key_bytes()})
    registry.publish(
        second,
        trusted_keys={second_provider.key_id: second_provider.public_key_bytes()},
    )
    reviewer = Ed25519SigningProvider(key_id="reviewer-key")
    review = SignedAgentReview.sign(
        agent_name="second-agent",
        agent_version="1.0.0",
        reviewer="Independent Lab",
        rating=5,
        eval_score=1.0,
        comment="Reproduced the eval suite.",
        signing_provider=reviewer,
        signed_at="2026-07-26T00:00:00+00:00",
    )
    registry.add_review(review, trusted_keys={reviewer.key_id: reviewer.public_key_bytes()})

    ranked = registry.ranked()

    assert [item.entry.name for item in ranked] == ["first-agent", "second-agent"]
    assert ranked[0].score == 0.9
    assert ranked[1].score == 0.72
    assert ranked[1].review_count == 1


def test_tampered_signed_review_is_rejected(tmp_path: Path) -> None:
    bundle, provider = _pack(tmp_path, "reviewer", 0.8)
    registry = FileAgentRegistry(tmp_path / "registry")
    registry.publish(bundle, trusted_keys={provider.key_id: provider.public_key_bytes()})
    reviewer = Ed25519SigningProvider(key_id="reviewer-key")
    review = SignedAgentReview.sign(
        agent_name="reviewer",
        agent_version="1.0.0",
        reviewer="Lab",
        rating=4,
        eval_score=0.8,
        signing_provider=reviewer,
    )
    tampered = replace(review, comment="Changed after signing")

    with pytest.raises(InvalidAgentBundleError, match="signature"):
        registry.add_review(
            tampered,
            trusted_keys={reviewer.key_id: reviewer.public_key_bytes()},
        )


def test_ranking_rechecks_stored_review_signature(tmp_path: Path) -> None:
    bundle, provider = _pack(tmp_path, "reviewer", 0.8)
    registry = FileAgentRegistry(tmp_path / "registry")
    registry.publish(bundle, trusted_keys={provider.key_id: provider.public_key_bytes()})
    reviewer = Ed25519SigningProvider(key_id="reviewer-key")
    review = SignedAgentReview.sign(
        agent_name="reviewer",
        agent_version="1.0.0",
        reviewer="Lab",
        rating=4,
        eval_score=0.8,
        signing_provider=reviewer,
    )
    review_path = registry.add_review(
        review,
        trusted_keys={reviewer.key_id: reviewer.public_key_bytes()},
    )
    stored = json.loads(review_path.read_text(encoding="utf-8"))
    stored["rating"] = 5
    review_path.write_text(json.dumps(stored), encoding="utf-8")

    with pytest.raises(InvalidAgentBundleError, match="Stored review signature"):
        registry.ranked()
