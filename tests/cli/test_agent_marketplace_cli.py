from __future__ import annotations

import base64
from pathlib import Path

import pytest

from synapsekit.audit.signer import Ed25519SigningProvider
from synapsekit.cli.main import main


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    (source / "evals").mkdir(parents=True)
    (source / "README.md").write_text("# CLI agent\n", encoding="utf-8")
    (source / "evals" / "suite.json").write_text("{}\n", encoding="utf-8")
    return source


def test_agent_cli_keygen_pack_verify_install_and_publish(tmp_path: Path, capsys) -> None:
    private_key = tmp_path / "publisher.key"
    public_key = tmp_path / "publisher.pub"
    main(
        [
            "agent",
            "keygen",
            str(private_key),
            "--public-key",
            str(public_key),
            "--key-id",
            "cli-publisher",
        ]
    )
    assert private_key.stat().st_size == 32

    bundle = tmp_path / "cli-agent.agent"
    main(
        [
            "agent",
            "pack",
            str(_source(tmp_path)),
            "--output",
            str(bundle),
            "--name",
            "cli-agent",
            "--agent-version",
            "1.0.0",
            "--author",
            "CLI Publisher",
            "--private-key",
            str(private_key),
            "--key-id",
            "cli-publisher",
            "--eval-score",
            "0.95",
        ]
    )
    trusted_key = f"cli-publisher:{public_key.read_text(encoding='ascii').strip()}"

    with pytest.raises(SystemExit) as verified:
        main(["agent", "verify", str(bundle), "--trusted-key", trusted_key])
    assert verified.value.code == 0

    unpacked = tmp_path / "unpacked"
    main(
        [
            "agent",
            "unpack",
            str(bundle),
            str(unpacked),
            "--trusted-key",
            trusted_key,
            "--require-trusted",
        ]
    )
    assert (unpacked / "README.md").is_file()

    install_root = tmp_path / "installed"
    main(
        [
            "agent",
            "install",
            str(bundle),
            "--install-root",
            str(install_root),
            "--trusted-key",
            trusted_key,
            "--require-trusted",
        ]
    )
    assert (install_root / "cli-agent" / "1.0.0" / "README.md").is_file()

    registry = tmp_path / "registry"
    main(
        [
            "agent",
            "publish",
            str(bundle),
            "--registry",
            str(registry),
            "--trusted-key",
            trusted_key,
            "--require-trusted",
        ]
    )
    assert (registry / "index.json").is_file()
    output = capsys.readouterr().out
    assert "Packed signed agent bundle" in output
    assert "Publisher: TRUSTED" in output
    assert "Published cli-agent 1.0.0" in output


def test_agent_cli_verify_can_require_publisher_trust(tmp_path: Path) -> None:
    provider = Ed25519SigningProvider(key_id="self-signed")
    source = _source(tmp_path)
    from synapsekit.marketplace import pack_agent

    bundle = pack_agent(
        source,
        tmp_path / "self-signed.agent",
        name="self-signed",
        version="1.0.0",
        author="Publisher",
        signing_provider=provider,
    )

    with pytest.raises(SystemExit) as result:
        main(["agent", "verify", str(bundle), "--require-trusted"])

    assert result.value.code == 1


def test_agent_cli_rejects_malformed_trusted_key(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="expected KEY_ID"):
        main(["agent", "verify", str(tmp_path / "missing.agent"), "--trusted-key", "bad"])


def test_agent_cli_public_key_is_raw_ed25519_base64(tmp_path: Path) -> None:
    private_key = tmp_path / "publisher.key"
    public_key = tmp_path / "publisher.pub"
    main(["agent", "keygen", str(private_key), "--public-key", str(public_key)])

    decoded = base64.b64decode(public_key.read_text(encoding="ascii").strip(), validate=True)

    assert len(decoded) == 32
