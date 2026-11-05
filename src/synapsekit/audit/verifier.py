"""Standalone bundle verifier — the other half of the open bundle spec.

This module intentionally only depends on the Python standard library
plus ``cryptography`` (for Ed25519 verification) — not on
:mod:`synapsekit.audit.trace` or :mod:`synapsekit.audit.signer`. That is
the whole point: a compliance auditor with no SynapseKit installation
can copy this one file plus ``pip install cryptography`` and independently
verify a bundle produced months or years ago, using only the bundle
format documented in :mod:`synapsekit.audit.bundle`.

Verification never "silently succeeds" — any structural problem,
cryptographic failure, or unexpected exception is captured as a
:class:`~synapsekit.audit.types.Verdict` other than ``MATCH`` rather than
swallowed. The verdict is deliberately three-valued rather than a bool:
``DRIFT`` means the evidence actively contradicts what's claimed (a
broken hash chain, an invalid signature, a tampered Merkle root);
``UNVERIFIABLE`` means there isn't enough evidence to reach a conclusion
either way (a corrupted bundle, an unsupported schema version, a key
this verifier has no way to check against). Collapsing that distinction
into a single boolean would make "we couldn't check" indistinguishable
from "we checked and it's wrong".

IMPORTANT — trust anchor: by default, :func:`verify` checks every
signature against the public key *embedded in the bundle's own
manifest*. That proves internal self-consistency (nothing was edited
after export) but NOT authenticity — anyone can generate a fully
self-consistent bundle from scratch with a freshly generated keypair.
For real non-repudiation, pass ``trusted_keys`` with the signer's public
key(s) obtained independently of the bundle (e.g. published out-of-band,
pinned in your own config) — signatures from any other key are then
rejected outright, regardless of what the manifest itself claims.
"""

from __future__ import annotations

import base64
import hashlib
import json
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .types import GENESIS_HASH, AuditRecord, Verdict, VerificationResult

if TYPE_CHECKING:
    from .metrics import AuditMetrics

SUPPORTED_SCHEMA_VERSIONS = {"1.2"}

REQUIRED_ENTRIES = ("manifest.json", "trace.jsonl", "hashes.merkle", "signatures.json")

#: Must match synapsekit.audit.merkle's domain separation exactly, or a
#: bundle produced by this package would fail its own verifier.
_NODE_DOMAIN = b"\x01"

#: Must match synapsekit.audit.merkle's leaf domain separation (RFC 6962
#: §2.1: 0x00 for leaves, 0x01 for internal nodes) exactly, or a bundle
#: produced by this package would fail its own verifier.
_LEAF_DOMAIN = b"\x00"

#: (verdict, message) — the atomic unit of evidence a check contributes.
Finding = tuple[Verdict, str]


def _hash_leaf(value: str) -> str:
    return hashlib.sha256(_LEAF_DOMAIN + bytes.fromhex(value)).hexdigest()


def _hash_pair(left: str, right: str) -> str:
    return hashlib.sha256(_NODE_DOMAIN + bytes.fromhex(left) + bytes.fromhex(right)).hexdigest()


def _largest_power_of_two_less_than(n: int) -> int:
    k = 1
    while k * 2 < n:
        k *= 2
    return k


def _merkle_root(leaves: list[str]) -> str:
    """RFC 6962 recursive-split root — never duplicates a leaf to pad odd counts."""
    n = len(leaves)
    if n == 0:
        return hashlib.sha256(b"").hexdigest()
    if n == 1:
        return leaves[0]
    k = _largest_power_of_two_less_than(n)
    return _hash_pair(_merkle_root(leaves[:k]), _merkle_root(leaves[k:]))


def _verify_ed25519(public_key: bytes, data: bytes, signature: bytes) -> bool:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(signature, data)
        return True
    except InvalidSignature:
        return False


def _verify_signature(*, algorithm: str, public_key: bytes, data: bytes, signature: bytes) -> bool:
    if algorithm == "ed25519":
        return _verify_ed25519(public_key, data, signature)
    raise ValueError(f"unsupported signing algorithm: {algorithm!r}")


def _normalize_strings(value: Any) -> Any:
    """Recursively apply Unicode NFC normalization to every string leaf.

    Must mirror :func:`synapsekit.audit.serializer._normalize_strings`
    exactly, or a record hashed by the producing package (which
    normalizes) would not reproduce here (which wouldn't), and every
    bundle containing non-NFC text would spuriously fail verification.
    Also unwraps ``MappingProxyType``/frozenset — ``AuditRecord.from_dict``
    deep-freezes the payload it reconstructs (see types.py), so a record
    loaded from a bundle carries frozen containers just like a live one.
    """
    from types import MappingProxyType

    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, dict | MappingProxyType):
        return {_normalize_strings(k): _normalize_strings(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_normalize_strings(v) for v in value]
    if isinstance(value, set | frozenset):
        return sorted((_normalize_strings(v) for v in value), key=repr)
    return value


def _canonical_json(value: Any) -> bytes:
    """Canonical serialization for values already made of plain JSON types
    (str/int/float/bool/None/dict/list) — used for record and manifest
    hashing. A record's ``timestamp`` is the one exception (a live
    ``datetime``), handled via ``default=``. Must match
    :func:`synapsekit.audit.serializer.canonical_json` byte-for-byte for
    any value both can serialize, or this standalone verifier would
    disagree with the hashes SynapseKit itself produces.
    """

    def default(obj: Any) -> Any:
        from datetime import date, datetime, timezone
        from types import MappingProxyType

        if isinstance(obj, datetime):
            return obj.astimezone(timezone.utc).isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        if isinstance(obj, MappingProxyType):
            return dict(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not canonically serializable")

    normalized = _normalize_strings(value)
    text = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=default,
    )
    return text.encode("utf-8")


def _payload_hash(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _record_hash(rec: AuditRecord) -> str:
    metadata = {
        "event_id": rec.event_id,
        "parent_id": rec.parent_id,
        "run_id": rec.run_id,
        "kind": rec.kind,
        "actor": rec.actor,
        "payload_hash": rec.payload_hash,
        "timestamp": rec.timestamp,
        "prev_hash": rec.prev_hash,
        "schema_version": rec.schema_version,
        "redaction_policy_hash": rec.redaction_policy_hash,
        "redaction_status": rec.redaction_status,
    }
    return hashlib.sha256(_canonical_json(metadata)).hexdigest()


@dataclass
class LoadedBundle:
    manifest: dict[str, Any]
    records: list[AuditRecord]
    hashes_doc: dict[str, Any]
    signatures_doc: list[dict[str, Any]]
    provenance_doc: Any = None
    provenance_present: bool = False


def load_bundle(path: str | Path) -> LoadedBundle:
    """Parse required bundle entries and optional provenance without checks."""
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        missing = [n for n in REQUIRED_ENTRIES if n not in names]
        if missing:
            raise ValueError(f"bundle is missing required entries: {missing}")

        manifest = json.loads(zf.read("manifest.json"))
        hashes_doc = json.loads(zf.read("hashes.merkle"))
        signatures_doc = json.loads(zf.read("signatures.json"))
        provenance_present = "provenance.json" in names
        provenance_doc = json.loads(zf.read("provenance.json")) if provenance_present else None

        trace_text = zf.read("trace.jsonl").decode("utf-8")
        records: list[AuditRecord] = []
        for line in trace_text.splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(AuditRecord.from_dict(json.loads(line)))

    return LoadedBundle(
        manifest=manifest,
        records=records,
        hashes_doc=hashes_doc,
        signatures_doc=signatures_doc,
        provenance_doc=provenance_doc,
        provenance_present=provenance_present,
    )


def _verify_merkle_proof(
    leaf: str, siblings: list[str], sibling_is_right: list[bool], root: str
) -> bool:
    node = leaf
    for sibling, is_right in zip(siblings, sibling_is_right, strict=True):
        node = _hash_pair(node, sibling) if is_right else _hash_pair(sibling, node)
    return node == root


def _resolve_public_key(
    key_id: str, manifest_keys: dict[str, Any], trusted_keys: dict[str, bytes] | None
) -> tuple[bytes | None, Finding | None]:
    """Return ``(public_key_bytes, finding)`` for a key_id — ``finding`` is set on failure.

    When ``trusted_keys`` is supplied, it is authoritative and
    exclusive: a key_id not present there is rejected even if the
    bundle's own manifest claims a public key for it — the whole point
    of pinning is to never trust key material sourced from the artifact
    being verified. A missing key is UNVERIFIABLE (we lack evidence),
    not DRIFT (evidence contradicting a claim) — see the module docstring.
    """
    if trusted_keys is not None:
        key_bytes = trusted_keys.get(key_id)
        if key_bytes is None:
            return None, (
                Verdict.UNVERIFIABLE,
                f"key_id {key_id!r} is not in the caller-supplied trusted key set",
            )
        return key_bytes, None
    info = manifest_keys.get(key_id)
    if info is None:
        return None, (Verdict.UNVERIFIABLE, f"no public key registered for key_id {key_id!r}")
    return base64.b64decode(info["public_key_b64"]), None


def _verify_manifest_signature(
    manifest: dict[str, Any], *, trusted_keys: dict[str, bytes] | None
) -> list[Finding]:
    """Authenticate manifest metadata (record_count, run_ids, created_at, keys, ...).

    Without this, an attacker could alter those fields freely — they
    aren't part of any hashed record or covered by a batch signature
    (which only signs the Merkle root of the trace) — without failing
    verification.
    """
    manifest_hash = manifest.get("manifest_hash")
    signatures = manifest.get("manifest_signatures")
    if not manifest_hash or not signatures:
        return [
            (
                Verdict.UNVERIFIABLE,
                "manifest is not signed (missing manifest_hash/manifest_signatures) — its metadata is unauthenticated",
            )
        ]

    core = {k: v for k, v in manifest.items() if k not in ("manifest_hash", "manifest_signatures")}
    recomputed = hashlib.sha256(_canonical_json(core)).hexdigest()
    if recomputed != manifest_hash:
        return [
            (
                Verdict.DRIFT,
                "manifest_hash does not match the manifest's own content — manifest metadata was tampered with",
            )
        ]

    manifest_keys = manifest.get("keys", {})
    last_finding: Finding | None = None
    for sig in signatures:
        public_key, finding = _resolve_public_key(sig["key_id"], manifest_keys, trusted_keys)
        if public_key is None:
            last_finding = finding
            continue
        try:
            ok = _verify_signature(
                algorithm=sig["algorithm"],
                public_key=public_key,
                data=bytes.fromhex(manifest_hash),
                signature=base64.b64decode(sig["signature_b64"]),
            )
        except Exception as exc:
            last_finding = (
                Verdict.UNVERIFIABLE,
                f"manifest signature verification raised an error: {exc}",
            )
            continue
        if ok:
            return []
        last_finding = (
            Verdict.DRIFT,
            f"manifest signature from key_id {sig['key_id']!r} is invalid",
        )

    return [
        last_finding
        or (Verdict.UNVERIFIABLE, "no valid manifest signature found from a recognized key")
    ]


def _verify_provenance_graph(provenance: Any) -> list[Finding]:
    """Check the PROV-JSON node maps and relation endpoints without imports."""
    if not isinstance(provenance, dict):
        return [(Verdict.UNVERIFIABLE, "provenance document must be a JSON object")]

    prefix = provenance.get("prefix")
    if not isinstance(prefix, dict) or prefix.get("prov") != "http://www.w3.org/ns/prov#":
        return [(Verdict.UNVERIFIABLE, "provenance document has no valid PROV namespace prefix")]

    def is_qualified_name(identifier: str) -> bool:
        if identifier.count(":") != 1:
            return False
        namespace, local_name = identifier.split(":", 1)
        return bool(namespace and local_name and namespace in prefix)

    node_maps: dict[str, dict[str, Any]] = {}
    for section in ("entity", "activity", "agent"):
        nodes = provenance.get(section)
        if not isinstance(nodes, dict):
            return [(Verdict.UNVERIFIABLE, f"provenance {section} map must be a JSON object")]
        if any(
            not isinstance(identifier, str)
            or not is_qualified_name(identifier)
            or not isinstance(attrs, dict)
            for identifier, attrs in nodes.items()
        ):
            return [
                (
                    Verdict.UNVERIFIABLE,
                    f"provenance {section} map has an invalid or unqualified node identifier",
                )
            ]
        node_maps[section] = nodes

    relation_fields = {
        "used": {"prov:activity": "activity", "prov:entity": "entity"},
        "wasGeneratedBy": {"prov:entity": "entity", "prov:activity": "activity"},
        "wasDerivedFrom": {
            "prov:generatedEntity": "entity",
            "prov:usedEntity": "entity",
        },
        "wasAssociatedWith": {"prov:activity": "activity", "prov:agent": "agent"},
    }
    findings: list[Finding] = []
    for relation_name, endpoints in relation_fields.items():
        relations = provenance.get(relation_name, {})
        if not isinstance(relations, dict):
            findings.append(
                (Verdict.UNVERIFIABLE, f"provenance {relation_name} map must be a JSON object")
            )
            continue
        for relation_id, relation in relations.items():
            if (
                not isinstance(relation_id, str)
                or not is_qualified_name(relation_id)
                or not isinstance(relation, dict)
            ):
                findings.append(
                    (Verdict.UNVERIFIABLE, f"provenance {relation_name} has an invalid relation")
                )
                continue
            for field, node_section in endpoints.items():
                target = relation.get(field)
                if not isinstance(target, str) or target not in node_maps[node_section]:
                    findings.append(
                        (
                            Verdict.UNVERIFIABLE,
                            f"provenance {relation_name} relation {relation_id!r} has a dangling {field}",
                        )
                    )
    return findings


def _verify_optional_provenance(loaded: LoadedBundle) -> list[Finding]:
    """Verify an optional provenance entry and its signed manifest digest."""
    if loaded.manifest.get("selective_disclosure"):
        if loaded.provenance_present or "provenance_hash" in loaded.manifest:
            return [
                (
                    Verdict.UNVERIFIABLE,
                    "provenance in a selective-disclosure bundle is not covered by a signed manifest",
                )
            ]
        return []

    digest = loaded.manifest.get("provenance_hash")
    if not loaded.provenance_present:
        if "provenance_hash" in loaded.manifest:
            return [
                (Verdict.DRIFT, "manifest declares provenance_hash but provenance.json is missing")
            ]
        return []
    if not isinstance(digest, str):
        return [(Verdict.DRIFT, "provenance.json is present but manifest has no provenance_hash")]

    try:
        recomputed = _payload_hash(loaded.provenance_doc)
    except Exception as exc:
        return [(Verdict.UNVERIFIABLE, f"provenance document could not be hashed: {exc}")]
    if recomputed != digest:
        return [(Verdict.DRIFT, "provenance_hash does not match provenance.json")]
    return _verify_provenance_graph(loaded.provenance_doc)


def _verify_record_self_consistency(rec: AuditRecord, i: int) -> list[Finding]:
    """payload_hash matches the payload, and the overall hash matches the metadata + payload_hash."""
    try:
        recomputed_payload_hash = _payload_hash(rec.payload)
    except Exception as exc:
        return [
            (
                Verdict.UNVERIFIABLE,
                f"record {i} ({rec.event_id}) payload could not be hashed: {exc}",
            )
        ]
    if recomputed_payload_hash != rec.payload_hash:
        return [
            (
                Verdict.DRIFT,
                f"record {i} ({rec.event_id}) payload_hash does not match its payload — tampered payload",
            )
        ]
    try:
        recomputed_hash = _record_hash(rec)
    except Exception as exc:
        return [(Verdict.UNVERIFIABLE, f"record {i} ({rec.event_id}) could not be hashed: {exc}")]
    if recomputed_hash != rec.hash:
        return [
            (
                Verdict.DRIFT,
                f"record {i} ({rec.event_id}) hash does not match its content — tampered record",
            )
        ]
    return []


def _verify_selective(
    loaded: LoadedBundle, *, trusted_keys: dict[str, bytes] | None
) -> list[Finding]:
    """Verify a selective-disclosure bundle via per-record Merkle proofs.

    A subset export can't satisfy the "recompute every leaf in the
    batch" check (most leaves are intentionally absent), so each kept
    record instead carries an inclusion proof against the *original*
    signed Merkle root. Chain contiguity (prev_hash linkage) is not
    checked here — by design, disclosure legitimately omits ancestors.
    """
    findings: list[Finding] = []
    proofs_by_event = {p["event_id"]: p for p in loaded.hashes_doc.get("proofs", [])}
    sig_by_range = {(s["start_index"], s["end_index"]): s for s in loaded.signatures_doc}
    manifest_keys = loaded.manifest.get("keys", {})

    for i, rec in enumerate(loaded.records):
        self_findings = _verify_record_self_consistency(rec, i)
        if self_findings:
            findings.extend(self_findings)
            continue

        proof = proofs_by_event.get(rec.event_id)
        if proof is None:
            findings.append(
                (
                    Verdict.UNVERIFIABLE,
                    f"record {i} ({rec.event_id}) has no Merkle proof in this selective bundle",
                )
            )
            continue
        if proof["leaf"] != _hash_leaf(rec.hash):
            findings.append(
                (
                    Verdict.DRIFT,
                    f"record {i} ({rec.event_id}) hash does not match the leaf its Merkle proof was issued for",
                )
            )
            continue

        signature = sig_by_range.get((proof["batch_start"], proof["batch_end"]))
        if signature is None:
            findings.append(
                (
                    Verdict.UNVERIFIABLE,
                    f"record {i} ({rec.event_id}) references a batch with no matching signature",
                )
            )
            continue
        if not _verify_merkle_proof(
            proof["leaf"], proof["siblings"], proof["sibling_is_right"], signature["merkle_root"]
        ):
            findings.append(
                (
                    Verdict.DRIFT,
                    f"record {i} ({rec.event_id}) Merkle proof does not resolve to the signed root",
                )
            )
            continue

        public_key, finding = _resolve_public_key(signature["key_id"], manifest_keys, trusted_keys)
        if public_key is None:
            assert (
                finding is not None
            )  # _resolve_public_key always sets one when public_key is None
            findings.append((finding[0], f"record {i} ({rec.event_id}): {finding[1]}"))
            continue
        try:
            ok = _verify_signature(
                algorithm=signature["algorithm"],
                public_key=public_key,
                data=bytes.fromhex(signature["merkle_root"]),
                signature=base64.b64decode(signature["signature_b64"]),
            )
        except Exception as exc:
            findings.append(
                (
                    Verdict.UNVERIFIABLE,
                    f"record {i} ({rec.event_id}) signature verification raised an error: {exc}",
                )
            )
            continue
        if not ok:
            findings.append(
                (
                    Verdict.DRIFT,
                    f"record {i} ({rec.event_id}) signature is invalid for key_id {signature['key_id']!r}",
                )
            )

    return findings


def _resolve_verdict(findings: list[Finding]) -> Verdict:
    verdicts = {f[0] for f in findings}
    if Verdict.DRIFT in verdicts:
        return Verdict.DRIFT
    if Verdict.UNVERIFIABLE in verdicts:
        return Verdict.UNVERIFIABLE
    return Verdict.MATCH


#: The explanation attached whenever an otherwise-clean bundle is capped
#: at UNVERIFIABLE because no key was pinned. Kept as a constant so tests
#: and the CLI can recognise this specific, expected cause.
UNPINNED_MESSAGE = (
    "no trusted key was pinned, so signatures were checked only against keys embedded "
    "in the bundle itself — this proves the bundle wasn't edited after export, but NOT "
    "who produced it. Pass trusted_keys (obtained independently of the bundle) for a "
    "MATCH. Treating as UNVERIFIABLE."
)


def _apply_trust_anchor_cap(
    verdict: Verdict, findings: list[Finding], *, trust_anchor: str
) -> tuple[Verdict, list[Finding]]:
    """Cap a would-be MATCH at UNVERIFIABLE when no key was pinned.

    Without ``trusted_keys``, every signature was verified against a
    public key read from the bundle's *own* manifest — a fully
    self-consistent forgery is indistinguishable from a genuine bundle
    (see the module docstring). Reporting that as MATCH invites callers
    to treat "the bundle didn't lie about itself" as "the bundle is
    authentic". So a clean, unpinned verification is downgraded to
    UNVERIFIABLE; DRIFT (active contradiction) and an already-UNVERIFIABLE
    verdict are left exactly as they are.
    """
    if trust_anchor == "none" and verdict is Verdict.MATCH:
        return Verdict.UNVERIFIABLE, [*findings, (Verdict.UNVERIFIABLE, UNPINNED_MESSAGE)]
    return verdict, findings


def _record_metrics(m: AuditMetrics, findings: list[Finding]) -> None:
    for _verdict, message in {(f[0], f[1]) for f in findings}:
        m.record_verification_failure(reason=_categorize(message))


_FAILURE_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("schema version", "schema_version"),
    ("provenance", "provenance"),
    ("prev_hash", "hash_chain"),
    ("hash does not match", "hash_chain"),
    ("parent_id", "parent_id"),
    ("merkle", "merkle"),
    ("signature", "signature"),
    ("manifest", "manifest"),
    ("record_count", "manifest"),
    ("trusted key set", "untrusted_key"),
)


def _categorize(error: str) -> str:
    for needle, category in _FAILURE_CATEGORIES:
        if needle.lower() in error.lower():
            return category
    return "other"


def verify(
    bundle_path: str | Path,
    *,
    trusted_keys: dict[str, bytes] | None = None,
    metrics: AuditMetrics | None = None,
) -> VerificationResult:
    """Verify a bundle end to end: schema, hash chain, Merkle roots, signatures, manifest.

    Checks (in order, all run even after early failures where possible so
    a single ``verify()`` call surfaces every problem at once):

    1. bundle is a well-formed zip with all four required entries
    2. manifest schema_version is one this verifier understands
    3. every record's payload_hash matches its payload, and its overall
       hash matches its metadata + payload_hash
    4. every record's prev_hash links to the previous record in its run
       (catches tampering, reordering, and deletion)
    5. parent_id references point to an earlier record in the same run
    6. per-batch leaf hashes + Merkle root match what's recorded
    7. per-batch signature verifies against a public key — pinned via
       ``trusted_keys`` if supplied, otherwise the manifest's own key
       registry (see the module docstring on why that's a weaker claim)
    8. the manifest itself is signed and its metadata (record_count,
       run_ids, created_at, key registry, ...) matches that signature
    9. when present, ``provenance.json`` matches the signed
       ``provenance_hash`` and all PROV relation endpoints resolve

    Returns a :class:`~synapsekit.audit.types.VerificationResult` whose
    ``verdict`` is ``MATCH``, ``DRIFT`` (evidence contradicts a claim —
    tampering), or ``UNVERIFIABLE`` (not enough evidence to decide —
    e.g. a corrupted bundle or a key this call wasn't given).

    Pass ``trusted_keys={key_id: public_key_bytes, ...}`` (sourced
    independently of the bundle) for real non-repudiation; without it,
    verification only proves internal self-consistency.
    """
    from .metrics import default_metrics

    m = metrics if metrics is not None else default_metrics
    trust_anchor = "pinned" if trusted_keys is not None else "none"

    try:
        loaded = load_bundle(bundle_path)
    except Exception as exc:
        m.record_verification_failure(reason="corrupted_bundle")
        return VerificationResult(
            verdict=Verdict.UNVERIFIABLE,
            errors=[f"failed to open bundle: {exc}"],
            trust_anchor=trust_anchor,
        )

    manifest = loaded.manifest
    records = loaded.records
    findings: list[Finding] = []

    schema_version = manifest.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        findings.append(
            (
                Verdict.UNVERIFIABLE,
                f"unsupported schema version {schema_version!r}; this verifier supports {sorted(SUPPORTED_SCHEMA_VERSIONS)}",
            )
        )
        # A schema we don't understand may have an incompatible record
        # layout — stop here rather than guess at further checks.
        _record_metrics(m, findings)
        return VerificationResult(
            verdict=_resolve_verdict(findings),
            errors=[f for _, f in findings],
            record_count=len(records),
            schema_version=schema_version,
            trust_anchor=trust_anchor,
        )

    expected_count = manifest.get("record_count")
    if expected_count is None:
        findings.append((Verdict.UNVERIFIABLE, "manifest is missing 'record_count'"))
    elif expected_count != len(records):
        findings.append(
            (
                Verdict.DRIFT,
                f"manifest declares {expected_count} records but trace.jsonl contains {len(records)} "
                "— records were added or removed after signing",
            )
        )

    if manifest.get("selective_disclosure"):
        # A selective-disclosure manifest has no manifest_hash/
        # manifest_signatures of its own (there's no signing key
        # available when a subset is re-exported) — instead it embeds
        # the original, still-validly-signed manifest verbatim.
        original_manifest = manifest.get("original_manifest")
        if original_manifest is None:
            findings.append(
                (
                    Verdict.UNVERIFIABLE,
                    "selective-disclosure manifest is missing 'original_manifest' — its metadata cannot be authenticated",
                )
            )
        else:
            findings.extend(
                _verify_manifest_signature(original_manifest, trusted_keys=trusted_keys)
            )
        findings.extend(_verify_optional_provenance(loaded))
        findings.extend(_verify_selective(loaded, trusted_keys=trusted_keys))
        verdict, findings = _apply_trust_anchor_cap(
            _resolve_verdict(findings), findings, trust_anchor=trust_anchor
        )
        _record_metrics(m, findings)
        return VerificationResult(
            verdict=verdict,
            errors=[f for _, f in findings],
            record_count=len(records),
            batch_count=len(
                {(p["batch_start"], p["batch_end"]) for p in loaded.hashes_doc.get("proofs", [])}
            ),
            schema_version=schema_version,
            trust_anchor=trust_anchor,
        )

    findings.extend(_verify_manifest_signature(manifest, trusted_keys=trusted_keys))
    findings.extend(_verify_optional_provenance(loaded))

    actual_run_ids = sorted({r.run_id for r in records})
    if manifest.get("run_ids") is not None and manifest.get("run_ids") != actual_run_ids:
        findings.append(
            (
                Verdict.DRIFT,
                "manifest 'run_ids' does not match the run_ids actually present in trace.jsonl",
            )
        )

    # --- hash chain, per run_id ---------------------------------------
    seen_event_ids: dict[str, set[str]] = {}
    last_hash_by_run: dict[str, str] = {}
    for i, rec in enumerate(records):
        run_seen = seen_event_ids.setdefault(rec.run_id, set())
        expected_prev = last_hash_by_run.get(rec.run_id, GENESIS_HASH)
        if rec.prev_hash != expected_prev:
            findings.append(
                (
                    Verdict.DRIFT,
                    f"record {i} ({rec.event_id}) prev_hash does not match the preceding record "
                    f"in run {rec.run_id} — chain broken, reordered, or a record was deleted",
                )
            )
        findings.extend(_verify_record_self_consistency(rec, i))

        if rec.parent_id is not None and rec.parent_id not in run_seen:
            findings.append(
                (
                    Verdict.DRIFT,
                    f"record {i} ({rec.event_id}) has parent_id {rec.parent_id!r} "
                    f"which does not reference an earlier record in run {rec.run_id}",
                )
            )

        run_seen.add(rec.event_id)
        last_hash_by_run[rec.run_id] = rec.hash

    # --- Merkle roots + signatures, per batch ---------------------------
    batches = loaded.hashes_doc.get("batches", [])
    manifest_keys = manifest.get("keys", {})
    sig_by_range = {(s["start_index"], s["end_index"]): s for s in loaded.signatures_doc}

    for batch in batches:
        start, end = batch["start_index"], batch["end_index"]
        slice_records = records[start : end + 1]
        actual_leaves = [_hash_leaf(r.hash) for r in slice_records]
        if actual_leaves != batch["leaves"]:
            findings.append(
                (
                    Verdict.DRIFT,
                    f"batch [{start}:{end}] leaf hashes do not match the recorded records (reordered or edited)",
                )
            )
            continue

        recomputed_root = _merkle_root(actual_leaves)
        if recomputed_root != batch["merkle_root"]:
            findings.append(
                (
                    Verdict.DRIFT,
                    f"batch [{start}:{end}] Merkle root does not match its leaves — invalid Merkle tree",
                )
            )

        signature = sig_by_range.get((start, end))
        if signature is None:
            findings.append(
                (Verdict.UNVERIFIABLE, f"batch [{start}:{end}] has no matching signature")
            )
            continue
        if signature["merkle_root"] != batch["merkle_root"]:
            findings.append(
                (
                    Verdict.DRIFT,
                    f"batch [{start}:{end}] signature covers a different Merkle root than hashes.merkle records",
                )
            )
            continue

        public_key, finding = _resolve_public_key(signature["key_id"], manifest_keys, trusted_keys)
        if public_key is None:
            assert (
                finding is not None
            )  # _resolve_public_key always sets one when public_key is None
            findings.append((finding[0], f"batch [{start}:{end}]: {finding[1]}"))
            continue

        try:
            ok = _verify_signature(
                algorithm=signature["algorithm"],
                public_key=public_key,
                data=bytes.fromhex(signature["merkle_root"]),
                signature=base64.b64decode(signature["signature_b64"]),
            )
        except Exception as exc:
            findings.append(
                (
                    Verdict.UNVERIFIABLE,
                    f"batch [{start}:{end}] signature verification raised an error: {exc}",
                )
            )
            continue
        if not ok:
            findings.append(
                (
                    Verdict.DRIFT,
                    f"batch [{start}:{end}] signature is invalid for key_id {signature['key_id']!r}",
                )
            )

    covered = sum(b["end_index"] - b["start_index"] + 1 for b in batches)
    if covered != len(records):
        findings.append(
            (
                Verdict.DRIFT,
                f"batches cover {covered} records but the trace contains {len(records)} — unsigned records exist",
            )
        )

    verdict, findings = _apply_trust_anchor_cap(
        _resolve_verdict(findings), findings, trust_anchor=trust_anchor
    )
    _record_metrics(m, findings)

    return VerificationResult(
        verdict=verdict,
        errors=[f for _, f in findings],
        record_count=len(records),
        batch_count=len(batches),
        schema_version=schema_version,
        trust_anchor=trust_anchor,
    )
