from __future__ import annotations


class GraphConfigError(Exception):
    """Raised during compile() for invalid graph configuration."""


class GraphRuntimeError(Exception):
    """Raised during graph execution for runtime failures."""
