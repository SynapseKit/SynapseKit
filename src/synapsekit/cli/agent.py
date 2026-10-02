"""``synapsekit agent`` commands."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from ..agents.self_improving import AgentEvolutionAuditLog


def run_agent(args: Any) -> None:
    subcommand = getattr(args, "agent_command", None)
    if subcommand == "inspect-evolution":
        _run_inspect_evolution(args)
        return
    if subcommand == "keygen":
        _run_keygen(args)
        return
    if subcommand == "pack":
        _run_pack(args)
        return
    if subcommand == "verify":
        _run_verify(args)
        return
    if subcommand == "unpack":
        _run_unpack(args)
        return
    if subcommand == "install":
        _run_install(args)
        return
    if subcommand == "publish":
        _run_publish(args)
        return
    raise SystemExit(
        "Missing agent subcommand. Use: inspect-evolution, keygen, pack, verify, unpack, "
        "install, or publish"
    )


def _parse_trusted_keys(raw: list[str] | None) -> dict[str, bytes] | None:
    if not raw:
        return None
    trusted: dict[str, bytes] = {}
    for entry in raw:
        key_id, separator, encoded = entry.partition(":")
        if not separator or not key_id or not encoded:
            raise SystemExit(f"invalid --trusted-key {entry!r}; expected KEY_ID:BASE64_PUBLIC_KEY")
        try:
            public_key = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise SystemExit(f"invalid base64 public key for {key_id!r}") from exc
        if len(public_key) != 32:
            raise SystemExit(f"Ed25519 public key for {key_id!r} must be 32 bytes")
        trusted[key_id] = public_key
    return trusted


def _run_keygen(args: Any) -> None:
    from ..audit.signer import Ed25519SigningProvider

    private_path = Path(args.private_key)
    if private_path.exists():
        raise SystemExit(f"Refusing to overwrite existing private key: {private_path}")
    provider = Ed25519SigningProvider(key_id=args.key_id)
    private_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_bytes(provider.private_key_bytes())
    private_path.chmod(0o600)
    public_key_b64 = base64.b64encode(provider.public_key_bytes()).decode("ascii")
    if args.public_key is not None:
        public_path = Path(args.public_key)
        if public_path.exists():
            private_path.unlink()
            raise SystemExit(f"Refusing to overwrite existing public key: {public_path}")
        public_path.parent.mkdir(parents=True, exist_ok=True)
        public_path.write_text(public_key_b64 + "\n", encoding="ascii")
    print(f"Generated publisher key {provider.key_id} at {private_path}")
    print(f"Trusted key: {provider.key_id}:{public_key_b64}")


def _run_pack(args: Any) -> None:
    from ..marketplace import pack_agent

    private_key = Path(args.private_key).read_bytes()
    if len(private_key) != 32:
        raise SystemExit("Ed25519 private-key file must contain exactly 32 raw bytes")
    output = pack_agent(
        args.source,
        args.output,
        name=args.name,
        version=args.agent_version,
        author=args.author,
        signing_provider=private_key,
        description=args.description,
        entrypoint=args.entrypoint,
        tags=args.tags,
        eval_score=args.eval_score,
        key_id=args.key_id,
    )
    print(f"Packed signed agent bundle: {output}")


def _run_verify(args: Any) -> None:
    from ..marketplace import verify_agent_bundle

    trusted_keys = _parse_trusted_keys(args.trusted_keys)
    result = verify_agent_bundle(args.bundle, trusted_keys=trusted_keys)
    accepted = result.integrity_valid and (result.trusted or not args.require_trusted)
    if args.output_format == "json":
        print(
            json.dumps(
                {
                    "integrity_valid": result.integrity_valid,
                    "trusted": result.trusted,
                    "bundle_sha256": result.bundle_sha256,
                    "publisher_key_id": (
                        result.manifest.publisher.key_id if result.manifest is not None else None
                    ),
                    "errors": result.errors,
                },
                indent=2,
            )
        )
    else:
        print(f"Bundle: {args.bundle}")
        print(f"Integrity: {'VALID' if result.integrity_valid else 'INVALID'}")
        print(f"Publisher: {'TRUSTED' if result.trusted else 'UNPINNED'}")
        for error in result.errors:
            print(f"  - {error}")
    raise SystemExit(0 if accepted else 1)


def _run_unpack(args: Any) -> None:
    from ..marketplace import unpack_agent

    manifest = unpack_agent(
        args.bundle,
        args.output,
        trusted_keys=_parse_trusted_keys(args.trusted_keys),
        require_trusted=args.require_trusted,
    )
    print(f"Unpacked {manifest.name} {manifest.version} to {Path(args.output)}")


def _run_install(args: Any) -> None:
    from ..marketplace import install_agent

    installed = install_agent(
        args.bundle,
        install_root=args.install_root,
        trusted_keys=_parse_trusted_keys(args.trusted_keys),
        require_trusted=args.require_trusted,
    )
    trust = "trusted publisher" if installed.trusted else "untrusted; sandbox required"
    print(f"Installed {installed.manifest.name} {installed.manifest.version} at {installed.path}")
    print(f"Execution policy: {trust}")


def _run_publish(args: Any) -> None:
    from ..marketplace import FileAgentRegistry

    trusted_keys = _parse_trusted_keys(args.trusted_keys)
    if args.require_trusted and args.allow_untrusted:
        raise SystemExit("--require-trusted and --allow-untrusted cannot be used together")
    if args.require_trusted and trusted_keys is None:
        raise SystemExit("--require-trusted needs at least one --trusted-key")
    entry = FileAgentRegistry(args.registry).publish(
        args.bundle,
        trusted_keys=trusted_keys,
        allow_untrusted=args.allow_untrusted,
    )
    print(f"Published {entry.name} {entry.version} to {Path(args.registry)}")


def _run_inspect_evolution(args: Any) -> None:
    audit_path = Path(args.audit_path)
    if not audit_path.exists():
        print(f"No evolution audit log found at {audit_path}")
        return

    log = AgentEvolutionAuditLog(audit_path)
    rows = [
        patch
        for patch in log.list(limit=args.limit)
        if args.agent_id in {patch.before.get("agent_id"), patch.after.get("agent_id")}
        or patch.metadata.get("agent_id") == args.agent_id
        or args.agent_id == "all"
    ]

    if args.output_format == "json":
        print(json.dumps([patch.to_dict() for patch in rows], indent=2, default=str))
        return

    if not rows:
        print(f"No evolution patches found for agent '{args.agent_id}'.")
        return

    print()
    print(f"{'Patch':<12} {'Status':<12} {'Type':<18} {'Score':<8} {'Rollout':<8} Description")
    print("-" * 92)
    for patch in rows:
        score = f"{patch.eval_score:.3f}" if patch.eval_score is not None else "N/A"
        print(
            f"{patch.patch_id[:12]:<12} "
            f"{patch.status:<12} "
            f"{patch.patch_type:<18} "
            f"{score:<8} "
            f"{patch.rollout_pct:<8.1f} "
            f"{patch.description}"
        )
    print()
