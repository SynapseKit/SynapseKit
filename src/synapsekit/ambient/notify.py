"""Windows toast notification backend for ambient interventions."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def notify_windows_toast(title: str, message: str, *, timeout: int = 10) -> bool:
    """Fire a Windows toast via ``plyer``.

    Returns ``False`` (and logs) on any failure — a broken notification
    backend must never take the daemon down.
    """

    try:
        from plyer import notification

        notification.notify(title=title, message=message, timeout=timeout)
        return True
    except Exception:
        logger.warning("ambient: notification failed", exc_info=True)
        return False
