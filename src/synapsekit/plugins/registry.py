"""Plugin registry for SynapseKit."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BasePlugin


class PluginRegistry:
    """Central registry for SynapseKit plugins.

    Usage::

        from synapsekit.plugins import registry

        registry.register(MyPlugin)
        await registry.load("my_plugin")
        info = registry.list_plugins()
        instance = registry.get("my_plugin")
        await registry.unload("my_plugin")
    """

    def __init__(self) -> None:
        # name -> plugin class (registered but not necessarily loaded)
        self._classes: dict[str, type[BasePlugin]] = {}
        # name -> loaded instance
        self._instances: dict[str, BasePlugin] = {}

    def register(self, plugin_cls: type[BasePlugin]) -> None:
        """Register a plugin class.

        Args:
            plugin_cls: A subclass of ``BasePlugin`` with a ``name`` class variable.

        Raises:
            ValueError: If the class does not define ``name``.
        """
        if not hasattr(plugin_cls, "name") or not plugin_cls.name:
            raise ValueError(
                f"Plugin class {plugin_cls.__qualname__!r} must define a non-empty 'name' attribute."
            )
        self._classes[plugin_cls.name] = plugin_cls

    async def load(self, name: str) -> BasePlugin:
        """Instantiate and load a registered plugin.

        Args:
            name: The plugin name (must have been registered first).

        Returns:
            The loaded plugin instance.

        Raises:
            KeyError: If no plugin with that name is registered.
        """
        if name not in self._classes:
            raise KeyError(
                f"No plugin registered under name {name!r}. "
                f"Available: {list(self._classes)}"
            )
        if name in self._instances:
            return self._instances[name]

        instance = self._classes[name]()
        await instance.on_load()
        self._instances[name] = instance
        return instance

    async def unload(self, name: str) -> None:
        """Call ``on_unload`` and remove a loaded plugin instance.

        Args:
            name: The plugin name to unload.

        Raises:
            KeyError: If the plugin is not currently loaded.
        """
        if name not in self._instances:
            raise KeyError(f"Plugin {name!r} is not currently loaded.")
        instance = self._instances.pop(name)
        await instance.on_unload()

    def list_plugins(self) -> list[dict[str, object]]:
        """Return info dicts for all registered plugins.

        Returns:
            List of ``{name, version, description, loaded}`` dicts.
        """
        result = []
        for name, cls in self._classes.items():
            result.append(
                {
                    "name": name,
                    "version": cls.version,
                    "description": cls.description,
                    "loaded": name in self._instances,
                }
            )
        return result

    def get(self, name: str) -> BasePlugin:
        """Return a loaded plugin instance by name.

        Args:
            name: The plugin name.

        Raises:
            KeyError: If the plugin is not loaded.
        """
        if name not in self._instances:
            raise KeyError(
                f"Plugin {name!r} is not loaded. Call ``await registry.load({name!r})`` first."
            )
        return self._instances[name]
