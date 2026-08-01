"""Human-in-the-loop approvals through the Live dashboard.

An agent can pause on a sensitive action and wait for a human to click
Approve/Deny in the browser::

    from synapsekit.live import request_approval

    if await request_approval("send_email", "to customer@acme.com"):
        await send_email(...)

Publishes an ``approval.request`` event (the dashboard renders Approve/Deny
buttons) and blocks until the browser POSTs a decision to ``/approve`` — or the
timeout elapses, in which case ``default`` is returned. When Live is disabled
(no dashboard to ask) it returns ``default`` immediately rather than hang.
"""

from __future__ import annotations

import asyncio
import threading
import uuid

from .bus import bus

_pending: dict[str, dict[str, object]] = {}
_lock = threading.Lock()


async def request_approval(
    action: str,
    detail: str = "",
    *,
    timeout: float = 300.0,
    default: bool = False,
) -> bool:
    """Ask a human to approve ``action`` in the dashboard; return their decision."""
    if not bus.enabled:
        return default

    approval_id = uuid.uuid4().hex[:12]
    event = threading.Event()
    with _lock:
        _pending[approval_id] = {"event": event, "approved": default}

    bus.publish(
        {
            "kind": "approval.request",
            "name": "approval.request",
            "id": approval_id,
            "status": "ok",
            "attributes": {"action": action, "detail": detail},
        }
    )

    granted = await asyncio.to_thread(event.wait, timeout)
    with _lock:
        record = _pending.pop(approval_id, None)
    approved = bool(record["approved"]) if (record and granted) else default

    bus.publish(
        {
            "kind": "approval.result",
            "name": "approval.result",
            "id": approval_id,
            "status": "ok" if approved else "error",
            "attributes": {"action": action, "approved": approved},
        }
    )
    return approved


def resolve(approval_id: str | None, approved: bool) -> bool:
    """Resolve a pending approval (called by the server's POST /approve)."""
    if not approval_id:
        return False
    with _lock:
        record = _pending.get(approval_id)
        if record is None:
            return False
        record["approved"] = bool(approved)
        event = record["event"]
    if isinstance(event, threading.Event):
        event.set()
    return True
