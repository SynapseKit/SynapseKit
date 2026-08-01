"""Bridge Python ``logging`` into the SynapseKit Live event bus.

Attaches a handler on the root logger that publishes each log record as a
``log`` event, so ``logger.info(...)`` / ``logger.error(...)`` calls from your
app (or from SynapseKit itself) show up live in the dashboard. No-op when Live
is disabled; no third-party dependencies.
"""

from __future__ import annotations

import contextlib
import logging

from .bus import bus


class LiveLogHandler(logging.Handler):
    """A logging handler that publishes records to the Live event bus."""

    def emit(self, record: logging.LogRecord) -> None:
        if not bus.enabled:
            return
        with contextlib.suppress(Exception):
            bus.publish(
                {
                    "kind": "log",
                    "level": record.levelname.lower(),
                    "logger": record.name,
                    "message": self.format(record),
                    "status": "error" if record.levelno >= logging.ERROR else "ok",
                }
            )


_attached = False


def attach_log_bridge(level: int = logging.INFO) -> None:
    """Route Python logging into the Live feed (idempotent).

    Lowers the root logger level to ``level`` only if it is currently higher, so
    INFO records actually reach the handler — without raising a level the user
    deliberately set lower.
    """
    global _attached
    if _attached:
        return
    _attached = True
    handler = LiveLogHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    effective = root.level or logging.WARNING
    if effective > level:
        root.setLevel(level)
