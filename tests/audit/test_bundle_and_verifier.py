"""Bundle export + standalone verification — positive and negative (tamper) cases."""

from __future__ import annotations

import base64
import json
from pathlib import Path

from synapsekit.audit import AuditTracer, EventKind, SigningPolicy, export_audit_bundle
from synapsekit.audit.provenance import source_id_hash
from synapsekit.audit.serializer import hash_value
from synapsekit.audit.types import Verdict
from synapsekit.audit.verifier import (
    LoadedBundle,
    _verify_optional_provenance,
    _verify_provenance_graph,
    verify,
)

from .conftest import (
    dump_trace_lines,
    load_trace_lines,
    manifest_keys_as_trusted,
    read_zip_entries,
    write_zip_entries,
)


class TestValidBundle:
    def test_valid_bundle_verifies_ok(self, bundle_path: Path):
        # A MATCH requires a pinned key now; pin the bundle's own advertised
        # keys since this test only asserts structural integrity.
        result = verify(bundle_path, trusted_keys=manifest_keys_as_trusted(bundle_path))
        assert result.ok
        assert result.errors == []
        assert result.record_count == 4

    def test_unpinned_valid_bundle_is_capped_at_unverifiable(self, bundle_path: Path):
        # Security cap: with no pinned key, an otherwise-valid bundle is
        # UNVERIFIABLE, not MATCH — self-consistency is not authenticity.
        from synapsekit.audit.types import Verdict

        result = verify(bundle_path)
        assert result.verdict == Verdict.UNVERIFIABLE
        assert not result.ok
        assert result.trust_anchor == "none"

    def test_batch_count_matches_number_of_signed_batches(self, tmp_path, sample_records):
        policy = SigningPolicy.ed25519()
        path = export_audit_bundle(sample_records, policy, tmp_path / "b.zip", batch_size=2)
        result = verify(path, trusted_keys=manifest_keys_as_trusted(path))
        assert result.ok
        assert result.batch_count == 2

    def test_empty_record_set_produces_a_verifiable_empty_bundle(self, tmp_path):
        policy = SigningPolicy.ed25519()
        path = export_audit_bundle([], policy, tmp_path / "empty.zip")
        result = verify(path, trusted_keys=manifest_keys_as_trusted(path))
        assert result.ok
        assert result.record_count == 0


class TestNegativeVerification:
    """Every listed tamper scenario MUST fail verification — never silently pass."""

    def _mutate(self, tmp_path: Path, bundle_path: Path, mutate_records) -> Path:
        entries = read_zip_entries(bundle_path)
        records = load_trace_lines(entries)
        records = mutate_records(records)
        dump_trace_lines(entries, records)
        out = tmp_path / "mutated.zip"
        write_zip_entries(out, entries)
        return out

    def test_remove_a_record(self, tmp_path, bundle_path):
        mutated = self._mutate(tmp_path, bundle_path, lambda recs: recs[:-1])
        result = verify(mutated)
        assert not result.ok

    def test_reorder_records(self, tmp_path, bundle_path):
        def swap(recs):
            recs = list(recs)
            recs[0], recs[1] = recs[1], recs[0]
            return recs

        mutated = self._mutate(tmp_path, bundle_path, swap)
        result = verify(mutated)
        assert not result.ok

    def test_modify_payload(self, tmp_path, bundle_path):
        def tamper(recs):
            recs[0]["payload"] = {**recs[0]["payload"], "prompt": "INJECTED"}
            return recs

        mutated = self._mutate(tmp_path, bundle_path, tamper)
        result = verify(mutated)
        assert not result.ok

    def test_modify_timestamp(self, tmp_path, bundle_path):
        def tamper(recs):
            recs[0]["timestamp"] = "2099-01-01T00:00:00+00:00"
            return recs

        mutated = self._mutate(tmp_path, bundle_path, tamper)
        result = verify(mutated)
        assert not result.ok

    def test_modify_hash(self, tmp_path, bundle_path):
        def tamper(recs):
            recs[0]["hash"] = "0" * 64
            return recs

        mutated = self._mutate(tmp_path, bundle_path, tamper)
        result = verify(mutated)
        assert not result.ok

    def test_invalid_parent_id(self, tmp_path, bundle_path):
        def tamper(recs):
            recs[1]["parent_id"] = "does-not-exist"
            return recs

        mutated = self._mutate(tmp_path, bundle_path, tamper)
        result = verify(mutated)
        assert not result.ok
        assert any("parent_id" in e for e in result.errors)

    def test_invalid_merkle_root(self, tmp_path, bundle_path):
        entries = read_zip_entries(bundle_path)
        hashes_doc = json.loads(entries["hashes.merkle"])
        hashes_doc["batches"][0]["merkle_root"] = "f" * 64
        entries["hashes.merkle"] = json.dumps(hashes_doc).encode("utf-8")
        out = tmp_path / "bad_merkle.zip"
        write_zip_entries(out, entries)
        result = verify(out)
        assert not result.ok

    def test_invalid_manifest_missing_record_count(self, tmp_path, bundle_path):
        entries = read_zip_entries(bundle_path)
        manifest = json.loads(entries["manifest.json"])
        del manifest["record_count"]
        entries["manifest.json"] = json.dumps(manifest).encode("utf-8")
        out = tmp_path / "bad_manifest.zip"
        write_zip_entries(out, entries)
        result = verify(out)
        assert not result.ok


class TestMerkleLeafDomainSeparation:
    """Regression test: the exported bundle's hashes.merkle leaves must be
    the RFC 6962 domain-separated leaf hashes (hash_leaf(record.hash)),
    not the raw record.hash values. On the buggy code, leaves ==
    [r.hash for r in records] -- indistinguishable from an internal node
    input. A valid bundle must still verify end to end with the fix.
    """

    def test_recorded_leaves_are_not_raw_record_hashes(self, bundle_path, sample_records):
        entries = read_zip_entries(bundle_path)
        hashes_doc = json.loads(entries["hashes.merkle"])
        recorded_leaves = hashes_doc["batches"][0]["leaves"]
        raw_hashes = [r.hash for r in sample_records]
        assert recorded_leaves != raw_hashes

    def test_recorded_leaves_match_hash_leaf_of_record_hash(self, bundle_path, sample_records):
        from synapsekit.audit.merkle import hash_leaf

        entries = read_zip_entries(bundle_path)
        hashes_doc = json.loads(entries["hashes.merkle"])
        recorded_leaves = hashes_doc["batches"][0]["leaves"]
        expected = [hash_leaf(r.hash) for r in sample_records]
        assert recorded_leaves == expected

    def test_valid_bundle_with_domain_separated_leaves_still_verifies(self, bundle_path):
        result = verify(bundle_path, trusted_keys=manifest_keys_as_trusted(bundle_path))
        assert result.ok
        assert result.errors == []

    def test_replaying_raw_record_hash_as_a_leaf_is_rejected(self, tmp_path, bundle_path):
        # Simulate an attacker who tries to pass a raw record.hash off as
        # a pre-domain-separated leaf -- must fail Merkle verification.
        entries = read_zip_entries(bundle_path)
        records = load_trace_lines(entries)
        hashes_doc = json.loads(entries["hashes.merkle"])
        hashes_doc["batches"][0]["leaves"][0] = records[0]["hash"]
        entries["hashes.merkle"] = json.dumps(hashes_doc).encode("utf-8")
        out = tmp_path / "raw_hash_as_leaf.zip"
        write_zip_entries(out, entries)
        result = verify(out)
        assert not result.ok

    def test_schema_version_mismatch(self, tmp_path, bundle_path):
        entries = read_zip_entries(bundle_path)
        manifest = json.loads(entries["manifest.json"])
        manifest["schema_version"] = "99.0"
        entries["manifest.json"] = json.dumps(manifest).encode("utf-8")
        out = tmp_path / "bad_schema.zip"
        write_zip_entries(out, entries)
        result = verify(out)
        assert not result.ok
        assert "unsupported schema version" in result.errors[0]

    def test_wrong_public_key(self, tmp_path, bundle_path):
        entries = read_zip_entries(bundle_path)
        manifest = json.loads(entries["manifest.json"])
        other = SigningPolicy.ed25519()
        for key_id in manifest["keys"]:
            manifest["keys"][key_id]["public_key_b64"] = base64.b64encode(
                other.provider.public_key_bytes()
            ).decode("ascii")
        entries["manifest.json"] = json.dumps(manifest).encode("utf-8")
        out = tmp_path / "wrong_key.zip"
        write_zip_entries(out, entries)
        result = verify(out)
        assert not result.ok

    def test_missing_zip_entry_is_reported_not_crashed(self, tmp_path, bundle_path):
        entries = read_zip_entries(bundle_path)
        del entries["signatures.json"]
        out = tmp_path / "missing_entry.zip"
        write_zip_entries(out, entries)
        result = verify(out)
        assert not result.ok
        assert result.errors  # never silently succeeds

    def test_corrupted_zip_is_reported_not_crashed(self, tmp_path):
        bad = tmp_path / "corrupt.zip"
        bad.write_bytes(b"not a zip file at all")
        result = verify(bad)
        assert not result.ok

    def test_missing_bundle_file_is_reported_not_crashed(self, tmp_path):
        result = verify(tmp_path / "does_not_exist.zip")
        assert not result.ok


class TestKeyRotation:
    def test_verification_succeeds_across_rotated_keys_in_one_bundle(
        self, tmp_path, sample_records
    ):
        key_a = SigningPolicy.ed25519(key_id="key-a")
        key_b = SigningPolicy.ed25519(key_id="key-b")
        batch_a = sample_records[:2]
        batch_b = sample_records[2:]

        path = export_audit_bundle(
            [],
            [(batch_a, key_a), (batch_b, key_b)],
            tmp_path / "rotated.zip",
        )
        result = verify(path, trusted_keys=manifest_keys_as_trusted(path))
        assert result.ok
        assert result.batch_count == 2

    def test_tampering_after_rotation_still_caught(self, tmp_path, sample_records):
        key_a = SigningPolicy.ed25519(key_id="key-a")
        key_b = SigningPolicy.ed25519(key_id="key-b")
        batch_a = sample_records[:2]
        batch_b = sample_records[2:]
        path = export_audit_bundle(
            [], [(batch_a, key_a), (batch_b, key_b)], tmp_path / "rotated2.zip"
        )

        entries = read_zip_entries(path)
        records = load_trace_lines(entries)
        records[3]["payload"] = {**records[3]["payload"], "decision": "TAMPERED"}
        dump_trace_lines(entries, records)
        out = tmp_path / "rotated2_tampered.zip"
        write_zip_entries(out, entries)

        result = verify(out)
        assert not result.ok


class TestProvenanceBundle:
    def test_signed_bundle_contains_hashed_provenance_json(self, tmp_path, sample_records):
        policy = SigningPolicy.ed25519()
        path = export_audit_bundle(sample_records, policy, tmp_path / "provenance.zip")
        entries = read_zip_entries(path)
        manifest = json.loads(entries["manifest.json"])
        provenance = json.loads(entries["provenance.json"])

        assert manifest["provenance_hash"] == hash_value(provenance)
        assert provenance["activity"]
        assert verify(
            path, trusted_keys={policy.provider.key_id: policy.provider.public_key_bytes()}
        ).ok

    def test_tampering_with_provenance_json_is_detected(self, tmp_path, sample_records):
        policy = SigningPolicy.ed25519()
        path = export_audit_bundle(sample_records, policy, tmp_path / "signed.zip")
        entries = read_zip_entries(path)
        provenance = json.loads(entries["provenance.json"])
        first_activity = next(iter(provenance["activity"].values()))
        first_activity["sk:kind"] = "TAMPERED"
        entries["provenance.json"] = json.dumps(provenance).encode("utf-8")
        tampered = tmp_path / "tampered_provenance.zip"
        write_zip_entries(tampered, entries)

        result = verify(
            tampered,
            trusted_keys={policy.provider.key_id: policy.provider.public_key_bytes()},
        )

        assert result.verdict == Verdict.DRIFT
        assert any("provenance" in error.lower() for error in result.errors)

    def test_bundle_without_optional_provenance_remains_verifiable(self, tmp_path, sample_records):
        policy = SigningPolicy.ed25519()
        path = export_audit_bundle(
            sample_records,
            policy,
            tmp_path / "legacy-shape.zip",
            include_provenance=False,
        )
        entries = read_zip_entries(path)
        manifest = json.loads(entries["manifest.json"])

        assert "provenance.json" not in entries
        assert "provenance_hash" not in manifest
        assert verify(
            path, trusted_keys={policy.provider.key_id: policy.provider.public_key_bytes()}
        ).ok

    def test_answer_cited_sources_match_signed_provenance_lineage(self, tmp_path):
        policy = SigningPolicy.ed25519()
        tracer = AuditTracer(run_id="rag-citations")
        tracer.record(
            EventKind.RETRIEVAL,
            {
                "output": [
                    {
                        "text": "private retrieved text",
                        "metadata": {"source_id": "document-17"},
                    }
                ]
            },
            actor="retriever",
            event_id="retrieval-event",
        )
        llm_call = tracer.record(
            EventKind.LLM_CALL,
            {
                "args": [[{"role": "user", "content": "private question"}]],
                "trusted_source_ids": ["document-17"],
            },
            actor="llm",
            event_id="answer-call",
        )
        tracer.record(
            EventKind.LLM_RESPONSE,
            {
                "output": {
                    "content": "Answer supported by [document-17]",
                    "citation_ids": ["document-17"],
                }
            },
            actor="llm",
            parent_id=llm_call.event_id,
            event_id="answer-event",
        )
        path = export_audit_bundle(tracer.records, policy, tmp_path / "cited.audit.zip")
        entries = read_zip_entries(path)
        provenance = json.loads(entries["provenance.json"])

        source_entity = next(
            entity_id
            for entity_id, entity in provenance["entity"].items()
            if entity.get("sk:sourceIdHash") == source_id_hash("document-17")
        )
        answer_entity = next(
            entity_id
            for entity_id, entity in provenance["entity"].items()
            if entity.get("sk:eventIdHash") == source_id_hash("answer-event")
        )
        assert any(
            relation["prov:generatedEntity"] == answer_entity
            and relation["prov:usedEntity"] == source_entity
            for relation in provenance["wasDerivedFrom"].values()
        )
        assert verify(
            path,
            trusted_keys={policy.provider.key_id: policy.provider.public_key_bytes()},
        ).ok

    def test_graph_validation_rejects_unqualified_prov_ids(self, sample_records):
        from synapsekit.audit import build_provenance

        provenance = build_provenance(sample_records)
        provenance["entity"]["not-qualified"] = {}

        findings = _verify_provenance_graph(provenance)

        assert findings
        assert any("qualified" in error.lower() for _verdict, error in findings)

    def test_graph_validation_rejects_dangling_relation_endpoints(self, sample_records):
        from synapsekit.audit import build_provenance

        provenance = build_provenance(sample_records)
        provenance["used"]["sk:broken-use"] = {
            "prov:activity": "sk:missing-activity",
            "prov:entity": "sk:missing-entity",
        }

        findings = _verify_provenance_graph(provenance)

        assert any("dangling" in error.lower() for _verdict, error in findings)

    def test_null_provenance_hash_without_entry_is_not_treated_as_legacy(self):
        loaded = LoadedBundle(
            manifest={"provenance_hash": None},
            records=[],
            hashes_doc={},
            signatures_doc=[],
        )

        findings = _verify_optional_provenance(loaded)

        assert findings
