"""Human-in-the-loop approvals in the Live dashboard.

    python examples/live_hitl.py

Opens http://127.0.0.1:7900 and, on a loop, asks you to Approve/Deny a
sensitive action (sending an email) right in the browser. The agent blocks on
``request_approval`` until you click — this is the dashboard acting as a control
panel, not just a viewer. Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio

from synapsekit.live import enable, request_approval


async def send_email(to: str) -> None:
    print(f"  ✓ email sent to {to}")


async def main() -> None:
    enable(open_browser=True)
    print("Open the tab and click Approve/Deny when prompted. Ctrl+C to stop.")
    while True:
        # blocks until you click Approve/Deny in the dashboard (or 60s timeout)
        if await request_approval("send_email", "to customer@acme.com", timeout=60, default=False):
            await send_email("customer@acme.com")
        else:
            print("  ✕ denied — not sending")
        await asyncio.sleep(2.0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
