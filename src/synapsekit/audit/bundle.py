"""Audit bundle export — a portable, independently verifiable zip archive.

Bundle format (open specification — see :mod:`synapsekit.audit.verifier`
for a reader that only needs stdlib + ``cryptography``, not SynapseKit):

    bundle.zip
    |-- manifest.json     schema_version, run metadata, key registry, signed
    |-- trace.jsonl       one canonical-JSON AuditRecord per line, in order
    |-- hashes.merkle     JSON: per-batch leaf hash lists + computed roots
    |-- signatures.json   JSON list of Signature dicts (one per batch)
    `-- provenance.json   optional W3C PROV-JSON activity/entity graph

A "batch" is a contiguous slice of ``trace.jsonl`` whose leaf hashes were
combined into one Merkle root and signed once — see
:mod:`synapsekit.audit.signer`. Multiple batches (each potentially signed
by a different key) support key rotation within a single bundle.

``manifest.json``'s own metadata (record_count, run_ids, created_at, the
key registry, and the optional ``provenance_hash``) is itself hashed and
signed by every key used in the export (``manifest_hash`` /
``manifest_signatures``) — a batch signature only covers its Merkle root,
so without this, an attacker could alter manifest metadata freely without
invalidating anything.

Zip entries use a fixed timestamp so that exporting the same records
twice produces byte-identical archives (aside from the inherently
time-varying ``created_at``/``signed_at`` content fields).
"""

from __future__ import annotations

import base64
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .merkle import MerkleHasher, hash_leaf
from .provenance import build_provenance
from .serializer import hash_value
from .signer import SigningPolicy
from .types import SCHEMA_VERSION, AuditRecord, Signature

if TYPE_CHECKING:
    from .metrics import AuditMetrics

BUNDLE_SCHEMA_VERSION = SCHEMA_VERSION

#: Fixed so re-exporting identical content produces byte-identical zips —
#: real value is irrelevant, only needs to be a valid DOS date/time.
_FIXED_ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)


def _write_zip_entry(zf: zipfile.ZipFile, name: str, data: str) -> None:
    info = zipfile.ZipInfo(name, date_time=_FIXED_ZIP_DATE_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    zf.writestr(info, data)


@dataclass
class _Batch:
    records: list[AuditRecord]
    signature: Signature
    leaves: list[str]


def _sign_batch(records: list[AuditRecord], policy: SigningPolicy, *, start_index: int) -> _Batch:
    leaves = [hash_leaf(r.hash) for r in records]
    root = MerkleHasher.root(leaves)
    signature = policy.sign_batch(
        root, start_index=start_index, end_index=start_index + len(records) - 1
    )
    return _Batch(records=records, signature=signature, leaves=leaves)


def _chunk(records: list[AuditRecord], batch_size: int | None) -> list[list[AuditRecord]]:
    if not batch_size or batch_size >= len(records):
        return [records] if records else []
    return [records[i : i + batch_size] for i in range(0, len(records), batch_size)]


def _sign_manifest(manifest: dict[str, Any], policies: list[SigningPolicy]) -> None:
    """Hash ``manifest``'s current content and sign it with every distinct
    policy used in this export, mutating ``manifest`` in place to add
    ``manifest_hash``/``manifest_signatures``. Must be called *after* all
    other manifest fields are set and *before* the manifest is written out.
    """
    manifest_hash = hash_value(manifest)
    seen: dict[str, SigningPolicy] = {}
    for policy in policies:
        seen.setdefault(policy.provider.key_id, policy)
    manifest["manifest_hash"] = manifest_hash
    manifest["manifest_signatures"] = [p.sign_manifest_hash(manifest_hash) for p in seen.values()]


def export_audit_bundle(
    records: list[AuditRecord],
    signing_policy: SigningPolicy | list[tuple[list[AuditRecord], SigningPolicy]],
    output_path: str | Path,
    *,
    batch_size: int | None = None,
    metrics: AuditMetrics | None = None,
    include_provenance: bool = True,
) -> Path:
    """Export ``records`` as a signed, portable audit bundle.

    Two calling conventions:

    - ``export_audit_bundle(records, policy, path, batch_size=100)`` —
      chunk ``records`` into batches of ``batch_size`` and sign each with
      the same policy.
    - ``export_audit_bundle([], [(batch1, policy_a), (batch2, policy_b)],
      path)`` — pass pre-split ``(records, policy)`` batches explicitly,
      e.g. to simulate/exercise key rotation across a single export.

    By default, a privacy-preserving PROV-JSON graph is included as
    ``provenance.json`` and its digest is covered by the signed manifest.
    Set ``include_provenance=False`` to omit this optional entry.
    """
    from .metrics import default_metrics

    m = metrics if metrics is not None else default_metrics

    try:
        if isinstance(signing_policy, SigningPolicy):
            batches_in = [(chunk, signing_policy) for chunk in _chunk(records, batch_size)]
        else:
            batches_in = signing_policy

        all_records: list[AuditRecord] = []
        batches: list[_Batch] = []
        index = 0
        for chunk, policy in batches_in:
            if not chunk:
                continue
            batch = _sign_batch(chunk, policy, start_index=index)
            batches.append(batch)
            all_records.extend(chunk)
            index += len(chunk)

        # Manifest metadata must be authenticated even for a zero-record
        # export (no batches ever got signed) — fall back to the
        # single-policy calling convention's key if that's what we have.
        manifest_policies = [policy for _, policy in batches_in if policy is not None]
        if not manifest_policies and isinstance(signing_policy, SigningPolicy):
            manifest_policies = [signing_policy]

        keys: dict[str, dict[str, str]] = {}
        for batch in batches:
            sig = batch.signature
            keys[sig.key_id] = {"algorithm": sig.algorithm, "public_key_b64": sig.public_key_b64}
        for policy in manifest_policies:
            keys.setdefault(
                policy.provider.key_id,
                {
                    "algorithm": policy.provider.algorithm,
                    "public_key_b64": base64.b64encode(policy.provider.public_key_bytes()).decode(
                        "ascii"
                    ),
                },
            )

        run_ids = sorted({r.run_id for r in all_records})
        manifest: dict[str, Any] = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "record_count": len(all_records),
            "batch_count": len(batches),
            "run_ids": run_ids,
            "keys": keys,
        }
        provenance_doc = build_provenance(all_records) if include_provenance else None
        if provenance_doc is not None:
            manifest["provenance_hash"] = hash_value(provenance_doc)
        _sign_manifest(manifest, manifest_policies)

        hashes_doc = {
            "batches": [
                {
                    "start_index": b.signature.start_index,
                    "end_index": b.signature.end_index,
                    "leaves": b.leaves,
                    "merkle_root": b.signature.merkle_root,
                }
                for b in batches
            ]
        }

        signatures_doc = [b.signature.to_dict() for b in batches]

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            _write_zip_entry(
                zf,
                "trace.jsonl",
                "\n".join(json.dumps(r.to_dict(), sort_keys=True) for r in all_records)
                + ("\n" if all_records else ""),
            )
            _write_zip_entry(zf, "hashes.merkle", json.dumps(hashes_doc, indent=2, sort_keys=True))
            _write_zip_entry(
                zf, "signatures.json", json.dumps(signatures_doc, indent=2, sort_keys=True)
            )
            if provenance_doc is not None:
                _write_zip_entry(
                    zf,
                    "provenance.json",
                    json.dumps(provenance_doc, indent=2, sort_keys=True),
                )
            _write_zip_entry(zf, "manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    except Exception:
        m.record_bundle_export(ok=False)
        raise

    m.record_bundle_export(ok=True)
    return output_path


def export_selective_bundle(
    source_bundle: str | Path,
    output_path: str | Path,
    *,
    kinds: list[str] | None = None,
    run_ids: list[str] | None = None,
    metrics: AuditMetrics | None = None,
) -> Path:
    """Re-export a subset of an existing bundle's records.

    Dropping most of a run's records would normally break the per-batch
    "recompute all leaves, recompute the root" check in
    :mod:`synapsekit.audit.verifier`, since it assumes every record in
    the signed range is present. Instead, each kept record carries its
    own Merkle *inclusion proof* against the original signed root
    (``signatures.json`` is copied through unchanged), so a verifier can
    confirm each disclosed record really was part of a signed batch
    without needing the omitted records at all — no re-signing required.

    The original, still-validly-signed manifest is preserved verbatim
    under ``original_manifest`` so its metadata (record_count, run_ids,
    created_at, key registry) remains authenticated; we have no signing
    key available at this point to re-sign a new manifest, so the
    selective-disclosure wrapper fields (``kept_event_ids``, the reduced
    ``record_count``) are themselves not separately signed — only which
    subset was chosen is unauthenticated, not the disclosed content
    itself (that's still fully covered by per-record Merkle proofs).
    """
    from .metrics import default_metrics
    from .verifier import load_bundle

    m = metrics if metrics is not None else default_metrics

    try:
        loaded = load_bundle(source_bundle)
        index_by_event = {r.event_id: i for i, r in enumerate(loaded.records)}
        batches = loaded.hashes_doc.get("batches", [])

        kept = loaded.records
        if kinds is not None:
            kept = [r for r in kept if r.kind in kinds]
        if run_ids is not None:
            kept = [r for r in kept if r.run_id in run_ids]

        proofs = []
        for rec in kept:
            gi = index_by_event[rec.event_id]
            batch = next(b for b in batches if b["start_index"] <= gi <= b["end_index"])
            local_index = gi - batch["start_index"]
            proof = MerkleHasher.proof(batch["leaves"], local_index)
            proofs.append(
                {
                    "event_id": rec.event_id,
                    "batch_start": batch["start_index"],
                    "batch_end": batch["end_index"],
                    "leaf": proof.leaf,
                    "siblings": proof.siblings,
                    "sibling_is_right": proof.sibling_is_right,
                }
            )

        hashes_doc = {"selective_disclosure": True, "proofs": proofs}
        manifest = {
            "schema_version": loaded.manifest.get("schema_version"),
            "record_count": len(kept),
            "selective_disclosure": True,
            "kept_event_ids": sorted(r.event_id for r in kept),
            "keys": loaded.manifest.get("keys", {}),
            "original_manifest": loaded.manifest,
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            _write_zip_entry(
                zf,
                "trace.jsonl",
                "\n".join(json.dumps(r.to_dict(), sort_keys=True) for r in kept)
                + ("\n" if kept else ""),
            )
            _write_zip_entry(zf, "hashes.merkle", json.dumps(hashes_doc, indent=2, sort_keys=True))
            _write_zip_entry(
                zf, "signatures.json", json.dumps(loaded.signatures_doc, indent=2, sort_keys=True)
            )
            _write_zip_entry(zf, "manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    except Exception:
        m.record_bundle_export(ok=False)
        raise

    m.record_bundle_export(ok=True)
    return output_path
