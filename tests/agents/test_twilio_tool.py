from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from synapsekit.agents.tools.twilio import TwilioTool


def _make_tool(**kw):
    defaults = {
        "account_sid": "ACtest123",
        "auth_token": "token456",
        "from_number": "+15550001111",
    }
    defaults.update(kw)
    return TwilioTool(**defaults)


def _mock_urlopen(sid="SMxxxx"):
    resp = MagicMock()
    resp.read.return_value = json.dumps({"sid": sid}).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestTwilioTool:
    @pytest.mark.asyncio
    async def test_send_sms(self):
        tool = _make_tool()
        with patch("urllib.request.urlopen", return_value=_mock_urlopen("SM123")):
            res = await tool.run(action="send_sms", to="+15559999999", body="test msg")
        assert not res.is_error
        assert "SMS sent" in res.output
        assert "SM123" in res.output

    @pytest.mark.asyncio
    async def test_send_whatsapp(self):
        tool = _make_tool()
        with patch("urllib.request.urlopen", return_value=_mock_urlopen("SM456")):
            res = await tool.run(action="send_whatsapp", to="+15559999999", body="wa test")
        assert not res.is_error
        assert "WhatsApp" in res.output

    @pytest.mark.asyncio
    async def test_missing_creds(self):
        tool = TwilioTool(account_sid="", auth_token="", from_number="")
        res = await tool.run(action="send_sms", to="+15559999999", body="hi")
        assert res.is_error
        assert "TWILIO_ACCOUNT_SID" in res.error

    @pytest.mark.asyncio
    async def test_missing_from_number(self):
        tool = TwilioTool(account_sid="AC123", auth_token="tok", from_number="")
        res = await tool.run(action="send_sms", to="+15559999999", body="hi")
        assert res.is_error
        assert "TWILIO_FROM_NUMBER" in res.error

    @pytest.mark.asyncio
    async def test_no_action(self):
        tool = _make_tool()
        res = await tool.run(to="+1555", body="x")
        assert res.is_error
        assert "No action" in res.error

    @pytest.mark.asyncio
    async def test_no_recipient(self):
        tool = _make_tool()
        res = await tool.run(action="send_sms", body="x")
        assert res.is_error
        assert "recipient" in res.error.lower() or "to" in res.error.lower()

    @pytest.mark.asyncio
    async def test_no_body(self):
        tool = _make_tool()
        res = await tool.run(action="send_sms", to="+1555")
        assert res.is_error
        assert "body" in res.error.lower()

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        tool = _make_tool()
        res = await tool.run(action="call", to="+1555", body="x")
        assert res.is_error
        assert "Unknown action" in res.error

    @pytest.mark.asyncio
    async def test_api_error_handled(self):
        tool = _make_tool()
        with patch("urllib.request.urlopen", side_effect=Exception("connection refused")):
            res = await tool.run(action="send_sms", to="+15559999999", body="hi")
        assert res.is_error
        assert "Twilio error" in res.error

    @pytest.mark.asyncio
    async def test_whatsapp_to_prefix_not_duplicated(self):
        """whatsapp: prefix on 'to' must not be doubled."""
        tool = _make_tool()
        captured = {}

        async def spy_post(payload):
            captured.update(payload)
            return {"sid": "SM999"}

        tool._post = spy_post
        await tool.run(action="send_whatsapp", to="whatsapp:+15559999999", body="hi")
        assert captured["To"] == "whatsapp:+15559999999"
        assert not captured["To"].startswith("whatsapp:whatsapp:")

    @pytest.mark.asyncio
    async def test_whatsapp_from_prefix_not_duplicated(self):
        """whatsapp: prefix on from_number must not be doubled."""
        tool = _make_tool(from_number="whatsapp:+15550001111")
        captured = {}

        async def spy_post(payload):
            captured.update(payload)
            return {"sid": "SM998"}

        tool._post = spy_post
        await tool.run(action="send_whatsapp", to="+15559999999", body="hi")
        assert captured["From"] == "whatsapp:+15550001111"
        assert not captured["From"].startswith("whatsapp:whatsapp:")

    def test_schema(self):
        tool = _make_tool()
        s = tool.schema()
        assert s["function"]["name"] == "twilio"
        props = s["function"]["parameters"]["properties"]
        assert "action" in props
        assert "to" in props
        assert "body" in props

    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("TWILIO_ACCOUNT_SID", "ACenv")
        monkeypatch.setenv("TWILIO_AUTH_TOKEN", "tokenv")
        monkeypatch.setenv("TWILIO_FROM_NUMBER", "+15550000000")
        tool = TwilioTool()
        assert tool._sid == "ACenv"
        assert tool._token == "tokenv"
        assert tool._from == "+15550000000"

    def test_security_warning_logged(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            _make_tool()
        assert any("TwilioTool instantiated" in r.message for r in caplog.records)
