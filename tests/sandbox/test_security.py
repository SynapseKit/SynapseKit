"""Negative security tests locking the sandbox's traversal, bundle-integrity,
and evaluation-gate guarantees against regression."""

from __future__ import annotations

import asyncio
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from synapsekit.sandbox import (
    CallableEvalGate,
    DiffBundle,
    SandboxSecurityError,
)
from synapsekit.sandbox.types import FileChange, FileChangeKind, normalize_relative_path


@pytest.mark.parametrize(
    "unsafe",
    [
        "../escape.txt",
        "project/../../etc/passwd",
        "/absolute/path",
        "with\x00null.txt",
        "./here.txt",
        "",
    ],
)
def test_normalize_relative_path_rejects_unsafe_inputs(unsafe: str) -> None:
    with pytest.raises(ValueError):
        normalize_relative_path(unsafe)


def test_normalize_relative_path_accepts_safe_nested_path() -> None:
    assert normalize_relative_path("project/src/main.py") == "project/src/main.py"


@pytest.mark.parametrize("unsafe", ["../evil.txt", "a/../../b.txt", "/etc/passwd", "x\x00y"])
def test_diff_bundle_construction_rejects_traversal_paths(unsafe: str) -> None:
    with pytest.raises(ValueError):
        DiffBundle(
            host_root="/tmp",
            base_fingerprint="base",
            sandbox_id="sandbox",
            changes=(FileChange(FileChangeKind.ADD, unsafe, payload=b"x"),),
        )


def test_diff_bundle_rejects_duplicate_paths() -> None:
    with pytest.raises(SandboxSecurityError):
        DiffBundle(
            host_root="/tmp",
            base_fingerprint="base",
            sandbox_id="sandbox",
            changes=(
                FileChange(FileChangeKind.ADD, "dup.txt", payload=b"one"),
                FileChange(FileChangeKind.ADD, "dup.txt", payload=b"two"),
            ),
        )


def test_diff_bundle_read_rejects_digest_tamper(tmp_path: Path) -> None:
    bundle = DiffBundle(
        host_root=str(tmp_path),
        base_fingerprint="base",
        sandbox_id="sandbox",
        changes=(FileChange(FileChangeKind.ADD, "file.txt", payload=b"payload"),),
    )
    path = tmp_path / "change.diff.zip"
    bundle.write(path)

    # Round-trips cleanly before tampering.
    assert DiffBundle.read(path).digest == bundle.digest

    with zipfile.ZipFile(path, "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
    manifest["digest"] = "0" * 64
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest).encode("utf-8"))

    with pytest.raises(SandboxSecurityError):
        DiffBundle.read(path)


def test_apply_rejects_symlink_parent_traversal(tmp_path: Path) -> None:
    root = tmp_path / "host"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = root / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable on this runner")

    # ``link/evil.txt`` passes path normalization (no ``..``/absolute tokens),
    # so only the apply-side symlink-parent check can stop it escaping ``root``.
    bundle = DiffBundle(
        host_root=str(root),
        base_fingerprint="base",
        sandbox_id="sandbox",
        changes=(FileChange(FileChangeKind.ADD, "link/evil.txt", payload=b"escaped"),),
    )
    receipt = SimpleNamespace(passed=True, diff_sha256=bundle.digest)

    async def scenario() -> None:
        with pytest.raises(SandboxSecurityError):
            await bundle.apply(receipt)

    asyncio.run(scenario())
    assert not (outside / "evil.txt").exists()


def test_apply_refuses_bundle_when_receipt_digest_mismatches(tmp_path: Path) -> None:
    root = tmp_path / "host"
    root.mkdir()
    bundle = DiffBundle(
        host_root=str(root),
        base_fingerprint="base",
        sandbox_id="sandbox",
        changes=(FileChange(FileChangeKind.ADD, "file.txt", payload=b"data"),),
    )
    # A passing receipt bound to a *different* diff must not authorize this one.
    receipt = SimpleNamespace(passed=True, diff_sha256="mismatched-digest")

    from synapsekit.sandbox import ApplyConflictError

    async def scenario() -> None:
        with pytest.raises(ApplyConflictError):
            await bundle.apply(receipt)

    asyncio.run(scenario())
    assert not (root / "file.txt").exists()


def test_callable_eval_gate_enforces_score_threshold() -> None:
    diff = DiffBundle(
        host_root="/tmp",
        base_fingerprint="base",
        sandbox_id="sandbox",
        changes=(),
    )
    environment = SimpleNamespace(session_id="sandbox")

    below = CallableEvalGate(lambda *_: {"passed": True, "score": 0.3}, threshold=0.5)
    at_or_above = CallableEvalGate(lambda *_: {"passed": True, "score": 0.9}, threshold=0.5)

    rejected = asyncio.run(below.evaluate(diff, environment))
    approved = asyncio.run(at_or_above.evaluate(diff, environment))

    # An explicit ``passed=True`` is still gated by the score threshold.
    assert rejected.passed is False
    assert rejected.score == 0.3
    assert approved.passed is True
    assert approved.diff_sha256 == diff.digest
