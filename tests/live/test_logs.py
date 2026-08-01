"""Python logging → Live event bus bridge."""

from __future__ import annotations

import logging

from synapsekit.live import bus
from synapsekit.live.logs import LiveLogHandler, attach_log_bridge


def test_log_records_publish_to_bus() -> None:
    attach_log_bridge()  # idempotent
    was = bus.enabled
    bus.enabled = True
    bus.clear()
    try:
        logging.getLogger("test.app").info("hello from the app")
        logging.getLogger("test.app").error("something broke")
    finally:
        bus.enabled = was
    logs = [e for e in bus.history() if e["kind"] == "log"]
    levels = {e["level"] for e in logs}
    assert any("hello from the app" in e["message"] for e in logs)
    assert "info" in levels and "error" in levels
    # error-level logs are flagged as error status for the Errors tab
    assert any(e["status"] == "error" for e in logs if e["level"] == "error")


def test_handler_is_noop_when_disabled() -> None:
    h = LiveLogHandler()
    bus.enabled = False
    bus.clear()
    h.emit(logging.LogRecord("x", logging.INFO, "f", 1, "msg", None, None))
    assert bus.history() == []
