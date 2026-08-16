"""Attestable Dream Mode audit-bundle signing (#935).

Real objects only — real Ed25519 keys persisted to ``tmp_path``, real bundles,
real ``verify()``. No mocks.
"""

from __future__ import annotations

import asyncio
import stat
from pathlib import Path

from synapsekit.audit import SigningPolicy, Verdict, verify
from synapsekit.dream import DreamConfig, DreamMode, PowerStatus, render_briefing


def _config(tmp_path: Path, **overrides: object) -> DreamConfig:
    return DreamConfig(
        schedule="02:00",
        state_path=tmp_path / "state.sqlite3",
        audit_dir=tmp_path / "audit",
        **overrides,  # type: ignore[arg-type]
    )


def _run(mode: DreamMode):
    return asyncio.run(
        mode.run_once(force=True, power=PowerStatus(plugged_in=True, battery_percent=100))
    )


def test_persisted_key_is_stable_across_instances(tmp_path: Path) -> None:
    # Two DreamMode instances sharing a state dir sign with the SAME key,
    # so bundles across nights carry one stable, pinnable identity.
    first = DreamMode(config=_config(tmp_path))
    key_id = first.signing_key_id
    first.close()

    key_file = tmp_path / "signing_key"
    assert key_file.exists()
    # Key file is created 0600 (owner-only).
    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600

    second = DreamMode(config=_config(tmp_path))
    try:
        assert second.signing_key_id == key_id
    finally:
        second.close()


def test_bundle_verifies_as_match_with_pinned_trusted_keys(tmp_path: Path) -> None:
    mode = DreamMode(config=_config(tmp_path))
    try:
        result = _run(mode)
        assert result.status == "completed"
        assert result.audit_path is not None
        assert result.audit_attestable is True
        assert result.audit_key_id == mode.signing_key_id

        # Pinned verification against the instance's own key → real MATCH.
        pinned = verify(result.audit_path, trusted_keys=mode.trusted_keys())
        assert pinned.verdict == Verdict.MATCH
    finally:
        mode.close()


def test_unpinned_verify_is_still_unverifiable(tmp_path: Path) -> None:
    # Persisting a key does NOT change unpinned verification: without a
    # pinned trusted key, a self-signed bundle stays UNVERIFIABLE by design.
    mode = DreamMode(config=_config(tmp_path))
    try:
        result = _run(mode)
        assert verify(result.audit_path).verdict == Verdict.UNVERIFIABLE
    finally:
        mode.close()


def test_briefing_marks_attestable_with_key_id(tmp_path: Path) -> None:
    mode = DreamMode(config=_config(tmp_path))
    try:
        result = _run(mode)
        briefing = render_briefing(result)
        assert "attestable" in briefing
        assert "NOT attestable" not in briefing
        assert mode.signing_key_id in briefing
    finally:
        mode.close()


def test_caller_signing_policy_is_respected(tmp_path: Path) -> None:
    # An explicit policy overrides the persisted-key default and is treated
    # as attestable (the caller owns a pinnable key). No key file is written.
    policy = SigningPolicy.ed25519()
    mode = DreamMode(config=_config(tmp_path), signing_policy=policy)
    try:
        assert mode.signing_policy is policy
        assert mode._attestable is True
        assert not (tmp_path / "signing_key").exists()
    finally:
        mode.close()


def test_non_persistable_key_falls_back_to_non_attestable(tmp_path: Path) -> None:
    # If the key can't be persisted (its parent is a file, not a dir), fall
    # back to an ephemeral key and mark the run non-attestable — surfaced in
    # the briefing so the user knows the bundle can't be pinned.
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory", encoding="utf-8")

    mode = DreamMode(config=_config(tmp_path, signing_key_path=blocker / "signing_key"))
    try:
        assert mode._attestable is False
        result = _run(mode)
        assert result.audit_attestable is False
        assert "NOT attestable" in render_briefing(result)
    finally:
        mode.close()
