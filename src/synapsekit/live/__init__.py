"""SynapseKit Live — a zero-dependency, real-time localhost view of a run.

Everything SynapseKit does (spans, LLM calls, retrieval, tool calls, cost) is
published to a single in-process :class:`EventBus`. When Live is enabled, a tiny
stdlib ``http.server`` streams those events over Server-Sent Events to a
single-file dashboard in the browser — no FastAPI/uvicorn/websockets, **no new
dependencies**, and a no-op when disabled.

Enable it any of these ways::

    export SYNAPSEKIT_LIVE=1          # auto-start on first published event
    from synapsekit.live import serve; serve()   # explicit, returns the URL
    synapsekit ui --live             # from the CLI

Bound to 127.0.0.1 and token-gated: nothing leaves the machine.
"""

from __future__ import annotations

import contextlib
import os

from .bus import EventBus, bus, publish
from .server import serve

__all__ = ["EventBus", "bus", "enable", "publish", "serve"]

_TRUTHY = {"1", "true", "yes", "on"}
_autostart_checked = False


def enable(*, open_browser: bool = True, quiet: bool = False) -> str:
    """Start the Live dashboard and silently turn on span instrumentation.

    Returns the dashboard URL. Enabling observe here (with a silent in-memory
    exporter) is what makes a normal run show up in the feed without any extra
    setup. Idempotent.
    """
    url = serve(open_browser=open_browser, quiet=quiet)
    try:  # enable span instrumentation so real runs stream, without console noise
        from ..observe.runtime import InMemoryExporter, configure, is_enabled

        if not is_enabled():
            configure(exporter=InMemoryExporter(), cost_tracking=True)
    except Exception:  # pragma: no cover - observe optional/edge cases
        pass
    return url


def _maybe_autostart() -> None:
    """Auto-start Live once if ``SYNAPSEKIT_LIVE`` is set. Cheap no-op otherwise."""
    global _autostart_checked
    if _autostart_checked:
        return
    _autostart_checked = True
    if os.environ.get("SYNAPSEKIT_LIVE", "").strip().lower() in _TRUTHY:
        # port in use / headless — fail quietly rather than break the user's run
        with contextlib.suppress(OSError):
            enable(open_browser=True)
