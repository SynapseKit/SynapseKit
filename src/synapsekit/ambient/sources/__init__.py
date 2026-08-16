"""Ambient source plugins."""

from __future__ import annotations

from .base import AmbientSourcePlugin
from .git import GitSourcePlugin
from .terminal import TerminalSourcePlugin

__all__ = [
    "AmbientSourcePlugin",
    "GitSourcePlugin",
    "TerminalSourcePlugin",
]
