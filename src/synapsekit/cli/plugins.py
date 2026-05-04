"""SynapseKit CLI — ``synapsekit plugin`` command."""

from __future__ import annotations

import argparse
import asyncio


def run_plugin(args: argparse.Namespace) -> None:
    """Dispatch plugin subcommands."""
    plugin_command = getattr(args, "plugin_command", None)

    if plugin_command == "list":
        _cmd_list()
    elif plugin_command == "load":
        asyncio.run(_cmd_load(args.path))
    elif plugin_command == "info":
        _cmd_info(args.name)
    else:
        print("Usage: synapsekit plugin {list,load,info}")
        raise SystemExit(1)


def _cmd_list() -> None:
    from synapsekit.plugins import registry

    plugins = registry.list_plugins()
    if not plugins:
        print("No plugins registered.")
        return

    col_w = [12, 10, 40, 8]
    header = (
        f"{'Name':<{col_w[0]}}  {'Version':<{col_w[1]}}  "
        f"{'Description':<{col_w[2]}}  {'Loaded':<{col_w[3]}}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)
    for p in plugins:
        desc = str(p["description"])
        if len(desc) > col_w[2]:
            desc = desc[: col_w[2] - 3] + "..."
        print(
            f"{p['name']!s:<{col_w[0]}}  {p['version']!s:<{col_w[1]}}  "
            f"{desc:<{col_w[2]}}  {'yes' if p['loaded'] else 'no':<{col_w[3]}}"
        )


async def _cmd_load(path: str) -> None:
    from synapsekit.plugins import load_plugin_from_path, registry

    plugin_cls = load_plugin_from_path(path)
    await registry.load(plugin_cls.name)
    print(f"Plugin {plugin_cls.name!r} loaded successfully from {path}.")


def _cmd_info(name: str) -> None:
    from synapsekit.plugins import registry

    plugins = {p["name"]: p for p in registry.list_plugins()}
    if name not in plugins:
        print(f"Plugin {name!r} not found. Run 'synapsekit plugin list' to see registered plugins.")
        raise SystemExit(1)

    p = plugins[name]
    print(f"Name:        {p['name']}")
    print(f"Version:     {p['version']}")
    print(f"Description: {p['description'] or '(none)'}")
    print(f"Loaded:      {'yes' if p['loaded'] else 'no'}")
