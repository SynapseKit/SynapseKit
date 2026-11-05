"""Persistent MCP tool-definition pinning and drift detection."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_STORE_VERSION = 1
_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_STORE_WRITE_LOCK = threading.Lock()


class ToolPinningError(RuntimeError):
    """Base exception for MCP tool-pinning failures."""


class ToolPinStoreError(ToolPinningError):
    """The persistent pin store is unreadable, invalid, or cannot be written."""


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """A reviewable snapshot of an MCP tool definition and its fingerprint."""

    name: str
    description: str | None
    input_schema: Any
    fingerprint: str


@dataclass(frozen=True, slots=True)
class ToolPinComparison:
    """Immutable classification of current definitions against saved pins."""

    unpinned_tools: tuple[ToolDefinition, ...]
    changed_tools: tuple[ToolDefinition, ...]
    removed_tool_names: tuple[str, ...]


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"MCP tool definition is not valid JSON: {exc}") from exc


def tool_fingerprint(name: str, description: str | None, input_schema: Any) -> str:
    """Return a stable SHA-256 fingerprint for a tool's public definition."""
    if not isinstance(name, str) or not name:
        raise ValueError("MCP tool name must be a non-empty string")
    if description is not None and not isinstance(description, str):
        raise ValueError("MCP tool description must be a string or None")
    definition = {
        "name": name,
        "description": description,
        "input_schema": input_schema,
    }
    return hashlib.sha256(_canonical_json(definition).encode("utf-8")).hexdigest()


def make_tool_definition(
    name: str,
    description: str | None,
    input_schema: Any,
) -> ToolDefinition:
    """Make a JSON-safe snapshot suitable for displaying and explicitly approving."""
    schema_snapshot = json.loads(_canonical_json(input_schema))
    fingerprint = tool_fingerprint(name, description, schema_snapshot)
    return ToolDefinition(
        name=name,
        description=description,
        input_schema=schema_snapshot,
        fingerprint=fingerprint,
    )


class ToolApprovalRequiredError(ToolPinningError):
    """Raised when current MCP tool definitions require explicit review and approval."""

    def __init__(
        self,
        pending_tools: list[ToolDefinition] | tuple[ToolDefinition, ...],
        *,
        message: str | None = None,
    ) -> None:
        self.pending_tools = tuple(pending_tools)
        default_message = (
            f"{len(self.pending_tools)} MCP tool(s) require explicit approval before use"
        )
        super().__init__(default_message if message is None else message)


class ToolDriftDetectedError(ToolPinningError):
    """Raised when pinned tool definitions, or the registered tool set, have changed."""

    def __init__(
        self,
        changed_tools: list[ToolDefinition] | tuple[ToolDefinition, ...],
        *,
        removed_tool_names: list[str] | tuple[str, ...] = (),
        pending_tools: list[ToolDefinition] | tuple[ToolDefinition, ...] | None = None,
    ) -> None:
        self.changed_tools = tuple(changed_tools)
        self.removed_tool_names = tuple(removed_tool_names)
        self.pending_tools = tuple(self.changed_tools if pending_tools is None else pending_tools)
        details = []
        if self.changed_tools:
            details.append(f"{len(self.changed_tools)} changed")
        if self.removed_tool_names:
            details.append(f"{len(self.removed_tool_names)} removed")
        super().__init__(f"MCP tool drift detected ({', '.join(details)})")


class ToolDriftWarning(UserWarning):
    """A report-only MCP tool-definition drift alert."""


class ToolPinStore:
    """Versioned JSON store with process-local serialized read-modify-write updates."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def get_pins(self, server_id: str) -> dict[str, str]:
        """Return the saved tool-name-to-fingerprint mapping for ``server_id``."""
        server_id = _validate_server_id(server_id)
        payload = self._read_all()
        return dict(payload["servers"].get(server_id, {}))

    def set_pins(self, server_id: str, fingerprints: dict[str, str]) -> None:
        """Replace the approved fingerprints for ``server_id`` atomically."""
        server_id = _validate_server_id(server_id)
        if not isinstance(fingerprints, dict):
            raise ValueError("tool fingerprints must be a dictionary")

        normalized: dict[str, str] = {}
        for name, fingerprint in fingerprints.items():
            if not isinstance(name, str) or not name:
                raise ValueError("tool names in the pin store must be non-empty strings")
            if not isinstance(fingerprint, str) or not _FINGERPRINT_PATTERN.fullmatch(fingerprint):
                raise ValueError(f"invalid SHA-256 fingerprint for MCP tool {name!r}")
            normalized[name] = fingerprint

        with _STORE_WRITE_LOCK:
            payload = self._read_all()
            payload["servers"][server_id] = normalized
            self._write_all(payload)

    def pin(self, server_id: str, definitions: Iterable[ToolDefinition]) -> None:
        """Atomically save fingerprints for reviewed tool definitions."""
        current = _definitions_by_name(definitions)
        self.set_pins(
            server_id,
            {name: definition.fingerprint for name, definition in current.items()},
        )

    def compare(self, server_id: str, definitions: Iterable[ToolDefinition]) -> ToolPinComparison:
        """Classify current definitions as unpinned, changed, or removed."""
        current = _definitions_by_name(definitions)
        pins = self.get_pins(server_id)
        return ToolPinComparison(
            unpinned_tools=tuple(
                definition for name, definition in current.items() if name not in pins
            ),
            changed_tools=tuple(
                definition
                for name, definition in current.items()
                if name in pins and pins[name] != definition.fingerprint
            ),
            removed_tool_names=tuple(sorted(set(pins) - set(current))),
        )

    def _read_all(self) -> dict[str, Any]:
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {"version": _STORE_VERSION, "servers": {}}
        except (OSError, UnicodeError) as exc:
            raise ToolPinStoreError(f"cannot read MCP tool pin store {self.path}: {exc}") from exc

        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise ToolPinStoreError(
                f"invalid JSON in MCP tool pin store {self.path}: {exc}"
            ) from exc

        if not isinstance(payload, dict) or set(payload) != {"version", "servers"}:
            raise ToolPinStoreError("invalid MCP tool pin store structure")
        version = payload["version"]
        if type(version) is not int or version != _STORE_VERSION:
            raise ToolPinStoreError(f"unsupported MCP tool pin store version: {version!r}")
        servers = payload["servers"]
        if not isinstance(servers, dict):
            raise ToolPinStoreError("MCP tool pin store 'servers' must be an object")

        for server_id, pins in servers.items():
            if not isinstance(server_id, str) or not server_id.strip():
                raise ToolPinStoreError("MCP tool pin store contains an invalid server_id")
            if not isinstance(pins, dict):
                raise ToolPinStoreError(f"pins for MCP server {server_id!r} must be an object")
            for name, fingerprint in pins.items():
                if not isinstance(name, str) or not name:
                    raise ToolPinStoreError("MCP tool pin store contains an invalid tool name")
                if not isinstance(fingerprint, str) or not _FINGERPRINT_PATTERN.fullmatch(
                    fingerprint
                ):
                    raise ToolPinStoreError(
                        f"MCP tool pin store contains an invalid fingerprint for {name!r}"
                    )
        return payload

    def _write_all(self, payload: dict[str, Any]) -> None:
        temporary_path: str | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = temporary.name
                json.dump(payload, temporary, sort_keys=True, indent=2, ensure_ascii=False)
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, self.path)
        except OSError as exc:
            raise ToolPinStoreError(f"cannot write MCP tool pin store {self.path}: {exc}") from exc
        finally:
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass


def _validate_server_id(server_id: str) -> str:
    if not isinstance(server_id, str) or not server_id.strip():
        raise ValueError("server_id must be a non-empty stable string")
    return server_id


def _definitions_by_name(
    definitions: Iterable[ToolDefinition],
) -> dict[str, ToolDefinition]:
    try:
        iterator = iter(definitions)
    except TypeError as exc:
        raise TypeError("tool definitions must be iterable") from exc

    normalized: dict[str, ToolDefinition] = {}
    for definition in iterator:
        if not isinstance(definition, ToolDefinition):
            raise TypeError("tool definitions must contain ToolDefinition values")
        snapshot = make_tool_definition(
            definition.name, definition.description, definition.input_schema
        )
        if snapshot.fingerprint != definition.fingerprint:
            raise ValueError(
                f"MCP tool definition fingerprint is inconsistent for {definition.name!r}"
            )
        if snapshot.name in normalized:
            raise ValueError(f"duplicate MCP tool definition name {snapshot.name!r}")
        normalized[snapshot.name] = snapshot
    return normalized
