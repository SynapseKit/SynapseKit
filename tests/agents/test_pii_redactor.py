"""Tests for PIIRedactor (v1.3.0)."""

from __future__ import annotations

import pytest

from synapsekit.agents.pii_redactor import PIIRedactor
from synapsekit.llm.base import BaseLLM, LLMConfig


class _MockLLM(BaseLLM):
    def __init__(self, response: str = "response"):
        super().__init__(LLMConfig(model="mock", api_key="test", provider="openai"))
        self._response = response

    async def stream(self, prompt, **kw):
        yield self._response

    async def generate(self, prompt, **kw):
        return self._response


class TestPIIRedactor:
    def test_redact_email(self):
        redactor = PIIRedactor(pii_types=["email"])
        result = redactor.redact("Contact john@example.com for details")
        assert "[EMAIL_1]" in result.redacted_text
        assert "john@example.com" not in result.redacted_text
        assert "email" in result.pii_types_found

    def test_redact_phone(self):
        redactor = PIIRedactor(pii_types=["phone"])
        result = redactor.redact("Call 555-123-4567")
        assert "[PHONE_1]" in result.redacted_text
        assert "555-123-4567" not in result.redacted_text

    def test_redact_multiple_types(self):
        redactor = PIIRedactor(pii_types=["email", "phone"])
        result = redactor.redact("Email a@b.com or call 555-123-4567")
        assert "[EMAIL_1]" in result.redacted_text
        assert "[PHONE_1]" in result.redacted_text
        assert sorted(result.pii_types_found) == ["email", "phone"]

    def test_same_value_deduplication(self):
        redactor = PIIRedactor(pii_types=["email"])
        result = redactor.redact("a@b.com and a@b.com again")
        assert result.redacted_text.count("[EMAIL_1]") == 2
        assert len(result.mapping) == 1

    def test_restore_mask_mode(self):
        redactor = PIIRedactor(pii_types=["email"], mode="mask")
        result = redactor.redact("Email a@b.com")
        assert result.mapping  # has mapping
        restored = redactor.restore(result.redacted_text, result.mapping)
        assert "a@b.com" in restored

    def test_restore_redact_mode_noop(self):
        redactor = PIIRedactor(pii_types=["email"], mode="redact")
        result = redactor.redact("Email a@b.com")
        assert result.mapping == {}  # no mapping in redact mode
        restored = redactor.restore(result.redacted_text, result.mapping)
        assert "[EMAIL_1]" in restored  # stays redacted

    def test_no_pii_passthrough(self):
        redactor = PIIRedactor()
        result = redactor.redact("No personal info here")
        assert result.redacted_text == "No personal info here"
        assert result.pii_types_found == []

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            PIIRedactor(mode="invalid")

    async def test_wrap_generate(self):
        redactor = PIIRedactor(pii_types=["email"], mode="mask")
        llm = _MockLLM(response="Got it, [EMAIL_1]!")
        restored, _redaction = await redactor.wrap_generate(llm, "My email is a@b.com")
        assert "a@b.com" in restored
        assert "[EMAIL_1]" not in restored
