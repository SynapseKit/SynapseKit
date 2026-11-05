"""PII redaction: PII removed, hashes deterministic, and signatures remain valid post-redaction."""

from __future__ import annotations

from synapsekit.audit import AuditTracer, EventKind, SigningPolicy, export_audit_bundle
from synapsekit.audit.redact import PIIRedactor
from synapsekit.audit.serializer import hash_value
from synapsekit.audit.verifier import verify


class TestPIIRedactor:
    def test_email_is_redacted(self):
        redactor = PIIRedactor()
        out = redactor.redact_text("contact me at alice@example.com please")
        assert "alice@example.com" not in out
        assert "[REDACTED:EMAIL]" in out

    def test_phone_is_redacted(self):
        redactor = PIIRedactor()
        out = redactor.redact_text("call 555-123-4567 now")
        assert "555-123-4567" not in out

    def test_ssn_is_redacted(self):
        redactor = PIIRedactor()
        out = redactor.redact_text("ssn: 123-45-6789")
        assert "123-45-6789" not in out
        assert "[REDACTED:SSN]" in out

    def test_credit_card_is_redacted(self):
        redactor = PIIRedactor()
        out = redactor.redact_text("card 4111111111111111 expires soon")
        assert "4111111111111111" not in out

    def test_ip_address_is_redacted(self):
        redactor = PIIRedactor()
        out = redactor.redact_text("connect to 192.168.1.100 now")
        assert "192.168.1.100" not in out

    def test_text_without_pii_is_unchanged(self):
        redactor = PIIRedactor()
        text = "just a normal sentence with no secrets"
        assert redactor.redact_text(text) == text

    def test_redact_payload_recurses_through_nested_structures(self):
        redactor = PIIRedactor()
        payload = {"user": {"email": "bob@example.com", "notes": ["call 555-987-6543"]}}
        redacted = redactor.redact_payload(payload)
        assert "bob@example.com" not in redacted["user"]["email"]
        assert "555-987-6543" not in redacted["user"]["notes"][0]

    def test_redaction_count_increments(self):
        redactor = PIIRedactor()
        redactor.redact_text("a@b.com and c@d.com")
        assert redactor.redaction_count == 2

    def test_pluggable_ml_detector_supplements_regex_detectors(self):
        def fake_ner(text: str):
            idx = text.find("Bob Smith")
            return [(idx, idx + len("Bob Smith"), "PERSON")] if idx >= 0 else []

        redactor = PIIRedactor(ml_detector=fake_ner)
        out = redactor.redact_text("My name is Bob Smith and my email is x@y.com")
        assert "Bob Smith" not in out
        assert "x@y.com" not in out


class TestCreditCardDetectorLuhnValidation:
    """Regression test: the raw regex (13-16 digits, optional separators)
    used to redact ANY numeric ID of that length — no Luhn check. That
    over-redacts ordinary IDs (order numbers, tracking numbers) that
    merely happen to be the right length. On the buggy code, a
    Luhn-invalid 16-digit run was still redacted; on the fixed code it
    is left alone.
    """

    def test_luhn_invalid_16_digit_id_is_not_redacted(self):
        # A 16-digit run that is NOT Luhn-valid (fails the checksum) —
        # e.g. a database/order ID that merely looks like a card number.
        redactor = PIIRedactor()
        non_card_id = "1234567890123456"  # fails Luhn
        out = redactor.redact_text(f"order id {non_card_id} confirmed")
        assert non_card_id in out
        assert "[REDACTED:CREDIT_CARD]" not in out

    def test_luhn_valid_card_number_is_still_redacted(self):
        redactor = PIIRedactor()
        # Standard Visa test number — Luhn-valid.
        card = "4111111111111111"
        out = redactor.redact_text(f"card {card} on file")
        assert card not in out
        assert "[REDACTED:CREDIT_CARD]" in out

    def test_luhn_valid_card_with_dashes_is_redacted(self):
        redactor = PIIRedactor()
        card_with_dashes = "4111-1111-1111-1111"
        out = redactor.redact_text(f"card {card_with_dashes} on file")
        assert "1111-1111-1111-1111" not in out
        assert "[REDACTED:CREDIT_CARD]" in out

    def test_luhn_invalid_13_digit_run_is_not_redacted(self):
        # Isolate the CreditCardDetector so PhoneDetector can't also
        # match a 10-digit substring of this run and mask the assertion.
        from synapsekit.audit.redact import CreditCardDetector

        redactor = PIIRedactor(detectors=[CreditCardDetector])
        non_card = "1234567890123"  # 13 digits, fails Luhn
        out = redactor.redact_text(f"tracking {non_card} shipped")
        assert non_card in out
        assert "[REDACTED:CREDIT_CARD]" not in out


class TestRedactionBeforeHashing:
    """Redaction must run BEFORE hashing — this is a documented, deliberate tradeoff."""

    def test_hashing_the_redacted_payload_is_deterministic(self):
        redactor = PIIRedactor()
        payload = {"note": "email me at test@example.com"}
        redacted_once = redactor.redact_payload(payload)
        redacted_twice = PIIRedactor().redact_payload(payload)
        assert hash_value(redacted_once) == hash_value(redacted_twice)

    def test_bundle_built_from_redacted_payloads_verifies_normally(self, tmp_path):
        redactor = PIIRedactor()
        tracer = AuditTracer()
        raw_payload = {"tool": "email_tool", "input": {"to": "alice@example.com"}, "output": "sent"}
        tracer.record(EventKind.TOOL_CALL, redactor.redact_payload(raw_payload))

        path = export_audit_bundle(
            tracer.drain(), SigningPolicy.ed25519(), tmp_path / "redacted.zip"
        )
        # MATCH requires a pinned key now; pin the bundle's own advertised
        # keys since this test only asserts that redaction doesn't break
        # structural verification.
        from .conftest import manifest_keys_as_trusted

        result = verify(path, trusted_keys=manifest_keys_as_trusted(path))
        assert result.ok

        from .conftest import read_zip_entries

        entries = read_zip_entries(path)
        assert b"alice@example.com" not in entries["trace.jsonl"]
        assert b"alice@example.com" not in entries["provenance.json"]
        assert b"sent" not in entries["provenance.json"]
