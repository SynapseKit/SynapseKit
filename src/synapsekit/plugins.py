"""Plugin system using ``importlib.metadata`` entry points."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Callable
from typing import Any

_ENTRY_POINT_GROUP = "synapsekit.plugins"


class PluginRegistry:
    """Discover and load plugins registered via the ``synapsekit.plugins`` entry point group.

    Third-party packages register plugins in their ``pyproject.toml``::

        [project.entry-points."synapsekit.plugins"]
        my_plugin = "my_package.plugin:register"

    Usage::

        registry = PluginRegistry()
        names = registry.discover()
        registry.load("my_plugin")
        # or
        registry.load_all()
    """

    def __init__(self) -> None:
        self._loaded: dict[str, Any] = {}

    def discover(self) -> list[str]:
        """Find all installed plugins. Returns a list of plugin names."""
        group_eps = importlib.metadata.entry_points(group=_ENTRY_POINT_GROUP)
        return [ep.name for ep in group_eps]

    def load(self, name: str) -> Any:
        """Load a single plugin by name.

        Calls the entry point's register function and caches the result.

        Returns:
            The value returned by the plugin's register function.

        Raises:
            KeyError: If the plugin is not found.
        """
        if name in self._loaded:
            return self._loaded[name]

        group_eps = list(importlib.metadata.entry_points(group=_ENTRY_POINT_GROUP, name=name))

        if not group_eps:
            raise KeyError(f"Plugin '{name}' not found in entry point group '{_ENTRY_POINT_GROUP}'")

        register_fn: Callable[..., Any] = group_eps[0].load()
        result = register_fn()
        self._loaded[name] = result
        return result

    def load_all(self) -> dict[str, Any]:
        """Load all discovered plugins.

        Returns:
            Dict mapping plugin name to register function return value.
        """
        for name in self.discover():
            if name not in self._loaded:
                self.load(name)
        return dict(self._loaded)

    @property
    def loaded(self) -> dict[str, Any]:
        """Dict of currently loaded plugins."""
        return dict(self._loaded)
