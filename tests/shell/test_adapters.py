from __future__ import annotations

from synapsekit.shell import all_adapters


def test_all_supported_shells_have_init_plugins() -> None:
    scripts = {adapter.name: adapter.init_script() for adapter in all_adapters()}

    assert set(scripts) == {"bash", "zsh", "fish", "powershell"}
    assert all(
        "synshell" in script and "synapsekit shell run" in script for script in scripts.values()
    )
