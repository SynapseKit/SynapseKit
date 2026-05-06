"""Tests for the _loop fast event loop installer."""

from __future__ import annotations

from synapsekit._loop import install_fast_loop


def test_install_fast_loop_no_error():
    """install_fast_loop should never raise, even without uvloop."""
    install_fast_loop()


def test_install_fast_loop_idempotent():
    """Calling install_fast_loop multiple times is safe."""
    install_fast_loop()
    install_fast_loop()
