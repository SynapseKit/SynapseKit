"""Tests for WolframAlphaTool."""

from __future__ import annotations

import types
from unittest.mock import Mock

import pytest

from synapsekit.agents.tools.wolfram import WolframAlphaTool


def _install_fake_wolframalpha(monkeypatch, client_mock: Mock) -> None:
    module = types.ModuleType("wolframalpha")
    module.Client = client_mock
    monkeypatch.setitem(__import__("sys").modules, "wolframalpha", module)


@pytest.mark.asyncio
class TestWolframAlphaTool:
    async def test_requires_api_key(self):
        tool = WolframAlphaTool()
        result = await tool.run(query="2+2")
        assert result.error == "No WOLFRAM_API_KEY configured."

    async def test_requires_query(self):
        tool = WolframAlphaTool(api_key="test-key")
        result = await tool.run()
        assert result.error == "No query provided."

    async def test_successful_query(self, monkeypatch):
        result_obj = Mock()
        result_obj.text = "4"

        response = Mock()
        response.results = [result_obj]

        client_instance = Mock()
        client_instance.query.return_value = response

        client_mock = Mock(return_value=client_instance)
        _install_fake_wolframalpha(monkeypatch, client_mock)

        tool = WolframAlphaTool(api_key="test-key")
        result = await tool.run(query="2+2")

        client_mock.assert_called_once_with("test-key")
        client_instance.query.assert_called_once_with("2+2")
        assert result.output == "4"
        assert result.error is None

    async def test_empty_results(self, monkeypatch):
        response = Mock()
        response.results = []

        client_instance = Mock()
        client_instance.query.return_value = response

        client_mock = Mock(return_value=client_instance)
        _install_fake_wolframalpha(monkeypatch, client_mock)

        tool = WolframAlphaTool(api_key="test-key")
        result = await tool.run(query="unknown")

        assert result.output == "No results found."

    async def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("WOLFRAM_API_KEY", "env-key")
        tool = WolframAlphaTool()
        assert tool._api_key == "env-key"

    async def test_exception_handling(self, monkeypatch):
        client_instance = Mock()
        client_instance.query.side_effect = Exception("Boom")

        client_mock = Mock(return_value=client_instance)
        _install_fake_wolframalpha(monkeypatch, client_mock)

        tool = WolframAlphaTool(api_key="test-key")
        result = await tool.run(query="2+2")
        assert "Wolfram Alpha error" in result.error
