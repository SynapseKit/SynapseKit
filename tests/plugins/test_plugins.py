"""Tests for the SynapseKit plugin system."""

from __future__ import annotations

import inspect
import textwrap
from pathlib import Path

import pytest

from synapsekit.plugins.base import BasePlugin
from synapsekit.plugins.registry import PluginRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _HelloPlugin(BasePlugin):
    name = "hello"
    version = "1.2.3"
    description = "A test plugin."

    on_load_called = False
    on_unload_called = False

    async def on_load(self) -> None:
        _HelloPlugin.on_load_called = True

    async def on_unload(self) -> None:
        _HelloPlugin.on_unload_called = True


# ---------------------------------------------------------------------------
# BasePlugin
# ---------------------------------------------------------------------------


def test_base_plugin_on_load_is_coroutine() -> None:
    assert inspect.iscoroutinefunction(BasePlugin.on_load)


def test_base_plugin_on_unload_is_coroutine() -> None:
    assert inspect.iscoroutinefunction(BasePlugin.on_unload)


def test_base_plugin_default_version() -> None:
    class _P(BasePlugin):
        name = "p"

    assert _P.version == "0.1.0"


def test_base_plugin_default_description() -> None:
    class _P(BasePlugin):
        name = "p"

    assert _P.description == ""


# ---------------------------------------------------------------------------
# PluginRegistry — register
# ---------------------------------------------------------------------------


def test_register_adds_plugin() -> None:
    reg = PluginRegistry()
    reg.register(_HelloPlugin)
    plugins = reg.list_plugins()
    assert any(p["name"] == "hello" for p in plugins)


def test_register_without_name_raises() -> None:
    class _NoName(BasePlugin):
        pass  # no name attribute

    reg = PluginRegistry()
    with pytest.raises((ValueError, AttributeError)):
        reg.register(_NoName)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# PluginRegistry — load / get / unload
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_calls_on_load() -> None:
    _HelloPlugin.on_load_called = False
    reg = PluginRegistry()
    reg.register(_HelloPlugin)
    instance = await reg.load("hello")
    assert _HelloPlugin.on_load_called is True
    assert isinstance(instance, _HelloPlugin)


@pytest.mark.asyncio
async def test_load_returns_same_instance_on_second_call() -> None:
    reg = PluginRegistry()
    reg.register(_HelloPlugin)
    inst1 = await reg.load("hello")
    inst2 = await reg.load("hello")
    assert inst1 is inst2


@pytest.mark.asyncio
async def test_load_unknown_plugin_raises() -> None:
    reg = PluginRegistry()
    with pytest.raises(KeyError):
        await reg.load("nonexistent")


@pytest.mark.asyncio
async def test_get_loaded_instance() -> None:
    reg = PluginRegistry()
    reg.register(_HelloPlugin)
    await reg.load("hello")
    instance = reg.get("hello")
    assert isinstance(instance, _HelloPlugin)


def test_get_unloaded_plugin_raises() -> None:
    reg = PluginRegistry()
    reg.register(_HelloPlugin)
    with pytest.raises(KeyError):
        reg.get("hello")


@pytest.mark.asyncio
async def test_unload_calls_on_unload() -> None:
    _HelloPlugin.on_unload_called = False
    reg = PluginRegistry()
    reg.register(_HelloPlugin)
    await reg.load("hello")
    await reg.unload("hello")
    assert _HelloPlugin.on_unload_called is True


@pytest.mark.asyncio
async def test_unload_removes_instance() -> None:
    reg = PluginRegistry()
    reg.register(_HelloPlugin)
    await reg.load("hello")
    await reg.unload("hello")
    with pytest.raises(KeyError):
        reg.get("hello")


@pytest.mark.asyncio
async def test_unload_unloaded_plugin_raises() -> None:
    reg = PluginRegistry()
    reg.register(_HelloPlugin)
    with pytest.raises(KeyError):
        await reg.unload("hello")


# ---------------------------------------------------------------------------
# list_plugins
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_plugins_loaded_flag() -> None:
    reg = PluginRegistry()
    reg.register(_HelloPlugin)
    before = {p["name"]: p for p in reg.list_plugins()}
    assert before["hello"]["loaded"] is False
    await reg.load("hello")
    after = {p["name"]: p for p in reg.list_plugins()}
    assert after["hello"]["loaded"] is True


def test_list_plugins_returns_version_and_description() -> None:
    reg = PluginRegistry()
    reg.register(_HelloPlugin)
    plugins = {p["name"]: p for p in reg.list_plugins()}
    assert plugins["hello"]["version"] == "1.2.3"
    assert plugins["hello"]["description"] == "A test plugin."


# ---------------------------------------------------------------------------
# load_plugin_from_path
# ---------------------------------------------------------------------------


def test_load_plugin_from_path_registers_and_returns(tmp_path: Path) -> None:
    plugin_file = tmp_path / "my_plugin.py"
    plugin_file.write_text(
        textwrap.dedent(
            """
            from synapsekit.plugins.base import BasePlugin

            class MyDynPlugin(BasePlugin):
                name = "my_dyn_plugin"
                version = "0.9.0"
                description = "Dynamic plugin."
            """
        )
    )
    from synapsekit.plugins import registry
    from synapsekit.plugins.loader import load_plugin_from_path

    cls = load_plugin_from_path(str(plugin_file))
    assert cls.name == "my_dyn_plugin"
    # Should be registered in global registry
    names = [p["name"] for p in registry.list_plugins()]
    assert "my_dyn_plugin" in names


def test_load_plugin_from_path_file_not_found() -> None:
    from synapsekit.plugins.loader import load_plugin_from_path

    with pytest.raises(FileNotFoundError):
        load_plugin_from_path("/nonexistent/path/plugin.py")


def test_load_plugin_from_path_no_subclass_raises(tmp_path: Path) -> None:
    plugin_file = tmp_path / "empty_plugin.py"
    plugin_file.write_text("x = 1\n")
    from synapsekit.plugins.loader import load_plugin_from_path

    with pytest.raises(ValueError, match="No BasePlugin subclass"):
        load_plugin_from_path(str(plugin_file))


# ---------------------------------------------------------------------------
# Global registry export
# ---------------------------------------------------------------------------


def test_global_registry_is_plugin_registry_instance() -> None:
    from synapsekit.plugins import registry

    assert isinstance(registry, PluginRegistry)


def test_exports() -> None:
    import synapsekit.plugins as pkg

    assert hasattr(pkg, "BasePlugin")
    assert hasattr(pkg, "PluginRegistry")
    assert hasattr(pkg, "load_plugin_from_path")
    assert hasattr(pkg, "registry")
