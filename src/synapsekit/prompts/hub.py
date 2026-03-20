"""Local filesystem prompt registry."""

from __future__ import annotations

import builtins
import json
import re
from pathlib import Path
from typing import Any

from .template import PromptTemplate

_DEFAULT_HUB_DIR = Path.home() / ".synapsekit" / "prompts"

# Ref format: "org/name:v2" or "org/name" (latest)
_REF_RE = re.compile(r"^(?P<org>[^/]+)/(?P<name>[^:]+)(?::(?P<version>.+))?$")


class PromptHub:
    """Local filesystem prompt registry.

    Layout: ``{hub_dir}/{org}/{name}/{version}.json``

    Usage::

        hub = PromptHub()
        hub.push("my-org/summarize", "Summarize: {text}", version="v1")
        tpl = hub.pull("my-org/summarize:v1")
        print(tpl.format(text="Hello world"))
    """

    def __init__(self, hub_dir: str | Path | None = None) -> None:
        self._hub_dir = Path(hub_dir) if hub_dir else _DEFAULT_HUB_DIR

    def push(
        self,
        name: str,
        template: str,
        version: str = "v1",
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save a prompt template to the local registry.

        Args:
            name: ``"org/prompt-name"`` format.
            template: The prompt template string.
            version: Version tag (e.g. ``"v1"``, ``"v2"``).
            metadata: Optional metadata dict.

        Returns:
            Path to the saved JSON file.
        """
        m = _REF_RE.match(f"{name}:{version}" if ":" not in name else name)
        if not m:
            raise ValueError(f"Invalid prompt name '{name}'. Expected format: 'org/name'")
        org, pname = m.group("org"), m.group("name")
        ver = m.group("version") or version

        prompt_dir = self._hub_dir / org / pname
        prompt_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "template": template,
            "version": ver,
            "metadata": metadata or {},
        }
        path = prompt_dir / f"{ver}.json"
        path.write_text(json.dumps(data, indent=2))
        return path

    def pull(self, ref: str) -> PromptTemplate:
        """Load a prompt template from the registry.

        Args:
            ref: ``"org/name:v2"`` or ``"org/name"`` (resolves to latest version).

        Returns:
            A ``PromptTemplate`` instance.
        """
        m = _REF_RE.match(ref)
        if not m:
            raise ValueError(
                f"Invalid prompt ref '{ref}'. Expected format: 'org/name' or 'org/name:version'"
            )
        org, name = m.group("org"), m.group("name")
        version = m.group("version")

        prompt_dir = self._hub_dir / org / name
        if not prompt_dir.exists():
            raise FileNotFoundError(f"Prompt '{org}/{name}' not found in hub")

        if version:
            path = prompt_dir / f"{version}.json"
            if not path.exists():
                raise FileNotFoundError(f"Version '{version}' of prompt '{org}/{name}' not found")
        else:
            # Resolve latest: sort version files and pick last
            versions = sorted(prompt_dir.glob("*.json"))
            if not versions:
                raise FileNotFoundError(f"No versions found for prompt '{org}/{name}'")
            path = versions[-1]

        data = json.loads(path.read_text())
        return PromptTemplate(data["template"])

    def list(self, org: str | None = None) -> builtins.list[str]:
        """List all prompts, optionally filtered by org.

        Returns:
            List of ``"org/name"`` strings.
        """
        results: list[str] = []
        if org:
            org_dir = self._hub_dir / org
            if org_dir.exists():
                for prompt_dir in sorted(org_dir.iterdir()):
                    if prompt_dir.is_dir():
                        results.append(f"{org}/{prompt_dir.name}")
        else:
            if self._hub_dir.exists():
                for org_dir in sorted(self._hub_dir.iterdir()):
                    if org_dir.is_dir():
                        for prompt_dir in sorted(org_dir.iterdir()):
                            if prompt_dir.is_dir():
                                results.append(f"{org_dir.name}/{prompt_dir.name}")
        return results

    def versions(self, name: str) -> builtins.list[str]:
        """List all versions of a prompt.

        Args:
            name: ``"org/prompt-name"`` format.

        Returns:
            List of version strings (e.g. ``["v1", "v2"]``).
        """
        m = _REF_RE.match(name + ":_")  # Add dummy version to match regex
        if not m:
            raise ValueError(f"Invalid prompt name '{name}'. Expected 'org/name'")
        org, pname = m.group("org"), m.group("name")

        prompt_dir = self._hub_dir / org / pname
        if not prompt_dir.exists():
            raise FileNotFoundError(f"Prompt '{name}' not found in hub")

        return sorted(p.stem for p in prompt_dir.glob("*.json"))
