"""SynapseKit plugin system."""

from __future__ import annotations

from .base import BasePlugin
from .loader import load_plugin_from_path
from .registry import PluginRegistry

#: Global plugin registry — use this to register and load plugins.
registry = PluginRegistry()

__all__ = [
    "BasePlugin",
    "PluginRegistry",
    "load_plugin_from_path",
    "registry",
]
