"""Base class for ambient source plugins."""

from __future__ import annotations

from synapsekit.plugins import BasePlugin

from ..events import AmbientEvent


class AmbientSourcePlugin(BasePlugin):
    """A ``BasePlugin`` that can be polled for new ``AmbientEvent``s."""

    async def poll(self) -> list[AmbientEvent]:
        raise NotImplementedError
