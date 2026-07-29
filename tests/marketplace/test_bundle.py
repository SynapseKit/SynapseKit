from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from synapsekit.audit.serializer import canonical_json
from synapsekit.audit.signer import Ed25519SigningProvider
from synapsekit.marketplace import (
    InvalidAgentBundleError,
    SandboxRequiredError,
    install_agent,
    pack_agent,
    unpack_agent,
    verify_agent_bundle,
)


def _agent_source(root: Path, *, marker: str = "v1") -> Path:
    source = root / "source"
    (source / "evals").mkdir(parents=True)
    (source / "memory").mkdir()
    (source / "router").mkdir()
    (source / "README.md").write_text("# PR reviewer\n", encoding="utf-8")
    (source / "agent.py").write_text(f"MARKER = {marker!r}\n", encoding="utf-8")
    (source / "evals" / "cases.json").write_text('{"cases": []}\n', encoding="utf-8")
    (source / "memory" / "preferences.md").write_text("Be concise.\n", encoding="utf-8")
    (source / "router" / "priors.json").write_text('{"model": "local"}\n', encoding="utf-8")
    return source


def _bundle(tmp_path: Path) -> tuple[Path, Ed25519SigningProvider]:
    provider = Ed25519SigningProvider(key_id="publisher-1")
    bundle = pack_agent(
        _agent_source(tmp_path),
        tmp_path / "reviewer.agent",
        name="pr-reviewer",
        version="1.2.0",
        author="Team Synapse",
        description="Reviews pull requests",
        entrypoint="agent.py:MARKER",
        tags=["review", "engineering"],
        eval_score=0.91,
        signing_provider=provider,
    )
    return bundle, provider


def _rewrite_archive(source: Path, output: Path, mutate) -> Path:
    with zipfile.ZipFile(source) as archive:
        entries = {info.filename: archive.read(info.filename) for info in archive.infolist()}
    mutate(entries)
    with zipfile.ZipFile(output, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return output


def _rewrite_signed_manifest(
    source: Path,
    output: Path,
    provider: Ed25519SigningProvider,
    mutate,
) -> Path:
    with zipfile.ZipFile(source) as archive:
        entries = {info.filename: archive.read(info.filename) for info in archive.infolist()}
    manifest = json.loads(entries["manifest.json"])
    mutate(manifest)
    entries["manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    entries["signature.ed25519"] = provider.sign(canonical_json(manifest))
    with zipfile.ZipFile(output, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return output


def test_pack_verify_and_unpack_round_trip(tmp_path: Path) -> None:
    bundle, provider = _bundle(tmp_path)
    trusted_keys = {provider.key_id: provider.public_key_bytes()}

    verification = verify_agent_bundle(bundle, trusted_keys=trusted_keys)

    assert verification.ok
    assert verification.trusted
    assert verification.manifest is not None
    assert verification.manifest.name == "pr-reviewer"
    assert verification.manifest.memory_format == "ump/1.0"
    assert verification.manifest.router_path == "router/"
    output = tmp_path / "unpacked"
    manifest = unpack_agent(bundle, output, trusted_keys=trusted_keys, require_trusted=True)
    assert manifest == verification.manifest
    assert (output / "memory" / "preferences.md").read_text(encoding="utf-8") == "Be concise.\n"
    assert (output / "signature.ed25519").stat().st_size == 64


def test_self_signed_bundle_has_integrity_but_not_identity_trust(tmp_path: Path) -> None:
    bundle, _ = _bundle(tmp_path)

    verification = verify_agent_bundle(bundle)

    assert verification.integrity_valid
    assert not verification.trusted
    assert verification.errors == ()


def test_same_inputs_create_byte_identical_bundle(tmp_path: Path) -> None:
    source = _agent_source(tmp_path)
    provider = Ed25519SigningProvider(key_id="deterministic")
    first = pack_agent(
        source,
        tmp_path / "first.agent",
        name="deterministic-agent",
        version="1.0.0",
        author="Publisher",
        signing_provider=provider,
    )
    second = pack_agent(
        source,
        tmp_path / "second.agent",
        name="deterministic-agent",
        version="1.0.0",
        author="Publisher",
        signing_provider=provider,
    )

    assert first.read_bytes() == second.read_bytes()


def test_payload_tampering_is_rejected(tmp_path: Path) -> None:
    bundle, _ = _bundle(tmp_path)

    tampered = _rewrite_archive(
        bundle,
        tmp_path / "tampered.agent",
        lambda entries: entries.__setitem__("agent.py", b"MARKER = 'pwned'\n"),
    )
    verification = verify_agent_bundle(tampered)

    assert not verification.ok
    assert any("Content hash mismatch" in error for error in verification.errors)


def test_untracked_archive_file_is_rejected(tmp_path: Path) -> None:
    bundle, _ = _bundle(tmp_path)

    tampered = _rewrite_archive(
        bundle,
        tmp_path / "extra.agent",
        lambda entries: entries.__setitem__("tools/hidden.py", b"print('unexpected')\n"),
    )
    verification = verify_agent_bundle(tampered)

    assert not verification.ok
    assert any("inventory mismatch" in error for error in verification.errors)


def test_path_traversal_archive_is_rejected_before_extraction(tmp_path: Path) -> None:
    bundle, _ = _bundle(tmp_path)

    tampered = _rewrite_archive(
        bundle,
        tmp_path / "traversal.agent",
        lambda entries: entries.__setitem__("../escape.py", b"bad\n"),
    )
    verification = verify_agent_bundle(tampered)

    assert not verification.ok
    assert any("Unsafe archive path" in error for error in verification.errors)
    assert not (tmp_path / "escape.py").exists()


def test_case_insensitive_path_collision_is_rejected(tmp_path: Path) -> None:
    bundle, _ = _bundle(tmp_path)
    tampered = _rewrite_archive(
        bundle,
        tmp_path / "case-collision.agent",
        lambda entries: entries.__setitem__("readme.md", b"shadow\n"),
    )

    verification = verify_agent_bundle(tampered)

    assert not verification.ok
    assert any("Case-insensitive" in error for error in verification.errors)


def test_windows_device_path_is_rejected_cross_platform(tmp_path: Path) -> None:
    bundle, _ = _bundle(tmp_path)
    tampered = _rewrite_archive(
        bundle,
        tmp_path / "device-name.agent",
        lambda entries: entries.__setitem__("tools/CON.py", b"shadow\n"),
    )

    verification = verify_agent_bundle(tampered)

    assert not verification.ok
    assert any("Unsafe archive path" in error for error in verification.errors)


def test_wrong_pinned_key_is_rejected(tmp_path: Path) -> None:
    bundle, provider = _bundle(tmp_path)
    other = Ed25519SigningProvider()

    verification = verify_agent_bundle(
        bundle, trusted_keys={provider.key_id: other.public_key_bytes()}
    )

    assert not verification.ok
    assert any("does not match" in error for error in verification.errors)


def test_untrusted_bundle_cannot_disable_mandatory_sandboxing(tmp_path: Path) -> None:
    bundle, provider = _bundle(tmp_path)
    unsafe = _rewrite_signed_manifest(
        bundle,
        tmp_path / "unsafe-sandbox-policy.agent",
        provider,
        lambda manifest: manifest["sandbox"].__setitem__("required_for_untrusted", False),
    )

    verification = verify_agent_bundle(unsafe)

    assert not verification.ok
    assert any("requires sandboxing" in error for error in verification.errors)


def test_pack_requires_evals_and_readme(tmp_path: Path) -> None:
    source = tmp_path / "incomplete"
    source.mkdir()
    (source / "README.md").write_text("# Incomplete\n", encoding="utf-8")

    with pytest.raises(ValueError, match="evals"):
        pack_agent(
            source,
            tmp_path / "incomplete.agent",
            name="incomplete",
            version="1.0.0",
            author="Publisher",
            signing_provider=Ed25519SigningProvider(),
        )


def test_pack_rejects_case_insensitive_reserved_install_metadata(tmp_path: Path) -> None:
    source = _agent_source(tmp_path)
    (source / ".SYNAPSEKIT-INSTALL.JSON").write_text("{}\n", encoding="utf-8")

    with pytest.raises(InvalidAgentBundleError, match="Reserved bundle path"):
        pack_agent(
            source,
            tmp_path / "reserved.agent",
            name="reserved",
            version="1.0.0",
            author="Publisher",
            signing_provider=Ed25519SigningProvider(),
        )


@pytest.mark.asyncio
async def test_untrusted_install_is_inert_and_uses_supplied_sandbox(tmp_path: Path) -> None:
    bundle, _ = _bundle(tmp_path)
    installed = install_agent(bundle, install_root=tmp_path / "installed")

    assert installed.sandbox_required
    assert (installed.path / ".synapsekit-install.json").is_file()
    with pytest.raises(SandboxRequiredError):
        await installed.run("review this")

    class RecordingSandbox:
        async def run(self, agent_path, manifest, prompt, **kwargs):
            return {
                "path": agent_path,
                "name": manifest.name,
                "prompt": prompt,
                "mode": kwargs["mode"],
            }

    result = await installed.run("review this", sandbox=RecordingSandbox(), mode="strict")
    assert result == {
        "path": installed.path,
        "name": "pr-reviewer",
        "prompt": "review this",
        "mode": "strict",
    }


def test_install_is_idempotent_for_same_bundle(tmp_path: Path) -> None:
    bundle, provider = _bundle(tmp_path)
    trusted_keys = {provider.key_id: provider.public_key_bytes()}

    first = install_agent(bundle, install_root=tmp_path / "installed", trusted_keys=trusted_keys)
    second = install_agent(bundle, install_root=tmp_path / "installed", trusted_keys=trusted_keys)

    assert first.path == second.path
    assert first.bundle_sha256 == second.bundle_sha256
    assert not first.sandbox_required
    metadata = json.loads((first.path / ".synapsekit-install.json").read_text(encoding="utf-8"))
    assert metadata["trusted"] is True


def test_invalid_component_cannot_escape_install_root(tmp_path: Path) -> None:
    source = _agent_source(tmp_path)
    with pytest.raises(ValueError, match="unsafe"):
        pack_agent(
            source,
            tmp_path / "unsafe.agent",
            name="../escape",
            version="1.0.0",
            author="Publisher",
            signing_provider=Ed25519SigningProvider(),
        )


def test_unpack_never_overwrites_an_existing_directory(tmp_path: Path) -> None:
    bundle, _ = _bundle(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError):
        unpack_agent(bundle, output)


def test_invalid_bundle_require_valid_raises_domain_error(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.agent"
    invalid.write_bytes(b"not a zip")

    verification = verify_agent_bundle(invalid)

    with pytest.raises(InvalidAgentBundleError):
        verification.require_valid()
