"""Base class for SynapseKit plugins."""

from __future__ import annotations

from abc import ABC
from typing import ClassVar


class BasePlugin(ABC):
    """Abstract base for SynapseKit plugins.

    Subclass and define ``name``, then optionally override
    ``on_load`` / ``on_unload``.

    Example::

        class MyPlugin(BasePlugin):
            name = "my_plugin"
            version = "1.0.0"
            description = "Does something useful."

            async def on_load(self) -> None:
                print("MyPlugin loaded")
    """

    name: ClassVar[str]
    version: ClassVar[str] = "0.1.0"
    description: ClassVar[str] = ""

    async def on_load(self) -> None:  # noqa: B027
        """Called when the plugin is loaded. Override to add setup logic."""

    async def on_unload(self) -> None:  # noqa: B027
        """Called when the plugin is unloaded. Override to add teardown logic."""
