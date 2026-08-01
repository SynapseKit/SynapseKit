"""Human-in-the-loop approvals through the Live bus."""

from __future__ import annotations

import asyncio

from synapsekit.live import bus, request_approval
from synapsekit.live.approvals import resolve


def test_returns_default_when_live_disabled() -> None:
    bus.enabled = False
    assert asyncio.run(request_approval("x", default=True)) is True
    assert asyncio.run(request_approval("x", default=False)) is False


def test_approval_granted_via_resolve() -> None:
    bus.enabled = True
    bus.clear()

    async def scenario() -> bool:
        # Approve as soon as the request event appears on the bus.
        async def approve_when_asked() -> None:
            for _ in range(200):
                reqs = [e for e in bus.history() if e["kind"] == "approval.request"]
                if reqs:
                    resolve(reqs[-1]["id"], True)
                    return
                await asyncio.sleep(0.01)

        task = asyncio.create_task(approve_when_asked())
        granted = await request_approval("send_email", "to a@b.com", timeout=5)
        await task
        return granted

    assert asyncio.run(scenario()) is True
    kinds = [e["kind"] for e in bus.history()]
    assert "approval.request" in kinds and "approval.result" in kinds


def test_approval_denied_via_resolve() -> None:
    bus.enabled = True
    bus.clear()

    async def scenario() -> bool:
        async def deny_when_asked() -> None:
            for _ in range(200):
                reqs = [e for e in bus.history() if e["kind"] == "approval.request"]
                if reqs:
                    resolve(reqs[-1]["id"], False)
                    return
                await asyncio.sleep(0.01)

        task = asyncio.create_task(deny_when_asked())
        granted = await request_approval("delete_all", timeout=5, default=False)
        await task
        return granted

    assert asyncio.run(scenario()) is False


def test_timeout_returns_default() -> None:
    bus.enabled = True
    bus.clear()
    # Nobody resolves it → default is returned quickly.
    assert asyncio.run(request_approval("noop", timeout=0.2, default=False)) is False
