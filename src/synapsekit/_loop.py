"""Optional fast event loop — installs ``uvloop`` if available."""

from __future__ import annotations


def install_fast_loop() -> None:
    """Install uvloop as the default asyncio event loop policy.

    No-op if uvloop is not installed or on unsupported platforms.
    """
    try:
        import uvloop

        uvloop.install()
    except (ImportError, AttributeError):
        pass
