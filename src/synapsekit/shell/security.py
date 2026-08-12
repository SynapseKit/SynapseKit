"""Ed25519 key helpers for shell pre-execution receipts."""

from __future__ import annotations

import base64
from contextlib import suppress
from pathlib import Path


def generate_signing_key(
    private_path: str | Path, public_path: str | Path | None = None
) -> dict[str, str]:
    """Generate a raw Ed25519 keypair for local shell audit receipts."""

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private_key = Ed25519PrivateKey.generate()
    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    private = Path(private_path).expanduser()
    private.parent.mkdir(parents=True, exist_ok=True)
    private.write_bytes(private_bytes)
    with suppress(OSError):
        private.chmod(0o600)
    public = (
        Path(public_path).expanduser()
        if public_path
        else private.with_suffix(private.suffix + ".pub")
    )
    public.write_text(base64.b64encode(public_bytes).decode("ascii") + "\n", encoding="utf-8")
    return {
        "private_key": str(private),
        "public_key": str(public),
        "key_id": base64.urlsafe_b64encode(public_bytes[:12]).decode("ascii").rstrip("="),
    }


def load_signing_policy(path: str | Path, *, key_id: str | None = None):
    """Load a raw Ed25519 private key into SynapseKit's audit policy."""

    from synapsekit.audit import SigningPolicy

    return SigningPolicy.ed25519(Path(path).expanduser().read_bytes(), key_id=key_id)
