"""Dynamic plugin loading from file paths."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BasePlugin


def load_plugin_from_path(path: str) -> type[BasePlugin]:
    """Load a ``BasePlugin`` subclass from a Python source file.

    Imports the file as a module, finds the first ``BasePlugin`` subclass
    defined in it, registers it in the global registry, and returns the class.

    Args:
        path: Absolute or relative path to a ``.py`` file containing a
              ``BasePlugin`` subclass.

    Returns:
        The first ``BasePlugin`` subclass found in the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If no ``BasePlugin`` subclass is found in the file.
    """
    from . import registry as _global_registry
    from .base import BasePlugin

    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Plugin file not found: {resolved}")

    module_name = f"_synapsekit_plugin_{resolved.stem}_{id(resolved)}"
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load plugin from {resolved}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    plugin_cls: type[BasePlugin] | None = None
    for _name, obj in inspect.getmembers(module, inspect.isclass):
        if obj is not BasePlugin and issubclass(obj, BasePlugin) and obj.__module__ == module_name:
            plugin_cls = obj
            break

    if plugin_cls is None:
        raise ValueError(
            f"No BasePlugin subclass found in {resolved}. "
            "Make sure your file defines a class that inherits from BasePlugin."
        )

    _global_registry.register(plugin_cls)
    return plugin_cls
