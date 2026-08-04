"""Tests for the Windows toast notifier wrapper."""

from __future__ import annotations

import sys
import types

from synapsekit.ambient.notify import notify_windows_toast


def test_notify_calls_plyer_with_expected_args(monkeypatch) -> None:
    calls = []

    fake_notification = types.SimpleNamespace(
        notify=lambda **kwargs: calls.append(kwargs)
    )
    fake_plyer = types.ModuleType("plyer")
    fake_plyer.notification = fake_notification
    monkeypatch.setitem(sys.modules, "plyer", fake_plyer)

    result = notify_windows_toast("Title", "Message", timeout=5)

    assert result is True
    assert calls == [{"title": "Title", "message": "Message", "timeout": 5}]


def test_notify_swallows_failure_and_returns_false(monkeypatch) -> None:
    def _raise(**kwargs):
        raise RuntimeError("no notification backend")

    fake_notification = types.SimpleNamespace(notify=_raise)
    fake_plyer = types.ModuleType("plyer")
    fake_plyer.notification = fake_notification
    monkeypatch.setitem(sys.modules, "plyer", fake_plyer)

    assert notify_windows_toast("Title", "Message") is False
