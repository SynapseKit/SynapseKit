"""Shell integration adapters and portable init scripts."""

from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod

from .types import ShellKind


class ShellAdapter(ABC):
    """Dialect-specific integration surface; execution remains argv-based."""

    kind: ShellKind

    @property
    def name(self) -> str:
        return self.kind.value

    @abstractmethod
    def init_script(self) -> str:
        """Return a script users can source in their current shell."""

    @abstractmethod
    def prompt(self) -> str:
        """Return the prompt marker used by the optional REPL."""


class BashAdapter(ShellAdapter):
    kind = ShellKind.BASH

    def init_script(self) -> str:
        return """# SynapseKit Agent OS Shell (bash)
synshell() { synapsekit shell run "$*"; }
alias synapse="synshell"
_synshell_complete() {
    local current="${COMP_WORDS[COMP_CWORD]}"
    COMPREPLY=( $(synapsekit shell complete --cwd "$PWD" "$current") )
}
complete -F _synshell_complete synshell
"""

    def prompt(self) -> str:
        return "synshell > "


class ZshAdapter(BashAdapter):
    kind = ShellKind.ZSH

    def init_script(self) -> str:
        return """# SynapseKit Agent OS Shell (zsh)
synshell() { synapsekit shell run "$*"; }
alias synapse="synshell"
_synshell_complete() { reply=($(synapsekit shell complete --cwd "$PWD" "$words[-1]")); }
compctl -K _synshell_complete synshell
"""


class FishAdapter(ShellAdapter):
    kind = ShellKind.FISH

    def init_script(self) -> str:
        return """# SynapseKit Agent OS Shell (fish)
function synshell
    synapsekit shell run (string join " " -- $argv)
end
alias synapse synshell
complete -c synshell -a '(synapsekit shell complete --cwd $PWD (commandline -ct))'
"""

    def prompt(self) -> str:
        return "synshell > "


class PowerShellAdapter(ShellAdapter):
    kind = ShellKind.POWERSHELL

    def init_script(self) -> str:
        return """# SynapseKit Agent OS Shell (PowerShell)
function synshell { synapsekit shell run ($args -join " ") }
Set-Alias synapse synshell
Register-ArgumentCompleter -CommandName synshell -ScriptBlock {
    param($commandName, $parameterName, $wordToComplete)
    synapsekit shell complete --cwd $pwd $wordToComplete | ForEach-Object {
        [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
    }
}
"""

    def prompt(self) -> str:
        return "synshell > "


_ADAPTERS: dict[ShellKind, ShellAdapter] = {
    ShellKind.BASH: BashAdapter(),
    ShellKind.ZSH: ZshAdapter(),
    ShellKind.FISH: FishAdapter(),
    ShellKind.POWERSHELL: PowerShellAdapter(),
}


def get_adapter(shell: ShellKind | str | None = None) -> ShellAdapter:
    if shell is None or str(shell).casefold() == "auto":
        shell = detect_shell()
    kind = shell if isinstance(shell, ShellKind) else ShellKind.parse(str(shell))
    return _ADAPTERS[kind]


def detect_shell() -> ShellKind:
    if os.name == "nt" or os.environ.get("PSMODULEPATH"):
        return ShellKind.POWERSHELL
    executable = os.path.basename(os.environ.get("SHELL", "")).casefold()
    if executable == "zsh":
        return ShellKind.ZSH
    if executable == "fish":
        return ShellKind.FISH
    if executable == "bash":
        return ShellKind.BASH
    return ShellKind.BASH if sys.platform != "win32" else ShellKind.POWERSHELL


def all_adapters() -> tuple[ShellAdapter, ...]:
    return tuple(_ADAPTERS.values())
