"""Deterministic, reviewable filesystem diff bundles."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import SandboxSecurityError
from .overlay import read_file_item
from .types import FileChange, FileChangeKind, SnapshotManifest, normalize_relative_path

_SCHEMA_VERSION = "1.0"
_FIXED_ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True, slots=True)
class DiffPreview:
    additions: int
    modifications: int
    deletions: int
    directories: int
    total_bytes: int

    @property
    def changes(self) -> int:
        return self.additions + self.modifications + self.deletions + self.directories


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _change_to_dict(change: FileChange, *, include_payload: bool) -> dict[str, Any]:
    data: dict[str, Any] = {
        "kind": change.kind.value,
        "path": normalize_relative_path(change.path),
        "before_sha256": change.before_sha256,
        "after_sha256": change.after_sha256,
        "size": change.size,
        "mode": change.mode,
    }
    if include_payload:
        data["payload"] = (
            base64.b64encode(change.payload).decode("ascii") if change.payload is not None else None
        )
    return data


def _change_from_dict(data: dict[str, Any]) -> FileChange:
    payload_value = data.get("payload")
    payload = base64.b64decode(payload_value) if payload_value is not None else None
    return FileChange(
        kind=FileChangeKind(str(data["kind"])),
        path=normalize_relative_path(str(data["path"])),
        before_sha256=data.get("before_sha256"),
        after_sha256=data.get("after_sha256"),
        size=int(data.get("size", 0)),
        mode=None if data.get("mode") is None else int(data["mode"]),
        payload=payload,
    )


@dataclass(frozen=True, slots=True)
class DiffBundle:
    """Immutable set of host-relative operations produced by a snapshot."""

    host_root: str
    base_fingerprint: str
    sandbox_id: str
    changes: tuple[FileChange, ...]
    audit_run_id: str | None = None
    created_at: str = ""
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"Unsupported diff bundle schema: {self.schema_version!r}.")
        if not self.created_at:
            object.__setattr__(self, "created_at", datetime.now(timezone.utc).isoformat())
        paths = [normalize_relative_path(change.path) for change in self.changes]
        if len(paths) != len(set(paths)):
            raise SandboxSecurityError("Diff bundle contains duplicate paths.")

    @classmethod
    def from_manifests(
        cls,
        before: SnapshotManifest,
        after: SnapshotManifest,
        *,
        current_root: str | Path,
        sandbox_id: str,
        audit_run_id: str | None = None,
    ) -> DiffBundle:
        before_by_path = {str(item["path"]): item for item in before.items}
        after_by_path = {str(item["path"]): item for item in after.items}
        changes: list[FileChange] = []
        for path in sorted(set(before_by_path) | set(after_by_path), key=str.casefold):
            old = before_by_path.get(path)
            new = after_by_path.get(path)
            if old == new:
                continue
            if old is None:
                changes.append(_make_change(FileChangeKind.ADD, path, new, current_root))
            elif new is None:
                kind = FileChangeKind.RMDIR if old.get("kind") == "dir" else FileChangeKind.DELETE
                changes.append(_make_change(kind, path, old, None))
            elif old.get("kind") != new.get("kind"):
                raise SandboxSecurityError(f"Filesystem type changes are not supported: {path!r}")
            elif new.get("kind") == "dir":
                continue
            elif new.get("kind") == "file":
                changes.append(_make_change(FileChangeKind.MODIFY, path, new, current_root, old))
            else:
                raise SandboxSecurityError(f"Symlink changes are not supported in diffs: {path!r}")
        changes.sort(key=_operation_sort_key)
        return cls(
            host_root=str(Path(before.root).resolve()),
            base_fingerprint=before.fingerprint,
            sandbox_id=sandbox_id,
            changes=tuple(changes),
            audit_run_id=audit_run_id,
        )

    @property
    def digest(self) -> str:
        return _sha256(_canonical(self.to_dict(include_payload=True)))

    def to_dict(self, *, include_payload: bool = True) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "host_root": self.host_root,
            "base_fingerprint": self.base_fingerprint,
            "sandbox_id": self.sandbox_id,
            "audit_run_id": self.audit_run_id,
            "created_at": self.created_at,
            "changes": [_change_to_dict(c, include_payload=include_payload) for c in self.changes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiffBundle:
        if str(data.get("schema_version")) != _SCHEMA_VERSION:
            raise ValueError("Unsupported diff bundle schema.")
        return cls(
            host_root=str(data["host_root"]),
            base_fingerprint=str(data["base_fingerprint"]),
            sandbox_id=str(data["sandbox_id"]),
            audit_run_id=data.get("audit_run_id"),
            created_at=str(data.get("created_at", "")),
            changes=tuple(_change_from_dict(value) for value in data.get("changes", [])),
        )

    def preview(self) -> DiffPreview:
        return DiffPreview(
            additions=sum(c.kind == FileChangeKind.ADD for c in self.changes),
            modifications=sum(c.kind == FileChangeKind.MODIFY for c in self.changes),
            deletions=sum(c.kind == FileChangeKind.DELETE for c in self.changes),
            directories=sum(
                c.kind in {FileChangeKind.MKDIR, FileChangeKind.RMDIR} for c in self.changes
            ),
            total_bytes=sum(c.size for c in self.changes if c.payload is not None),
        )

    def write(self, path: str | Path) -> Path:
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        manifest = self.to_dict(include_payload=True)
        manifest["digest"] = self.digest
        info = zipfile.ZipInfo("manifest.json", date_time=_FIXED_ZIP_DATE_TIME)
        info.compress_type = zipfile.ZIP_DEFLATED
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr(info, _canonical(manifest) + b"\n")
        return output

    @classmethod
    def read(cls, path: str | Path) -> DiffBundle:
        with zipfile.ZipFile(Path(path), "r") as archive:
            data = json.loads(archive.read("manifest.json"))
        bundle = cls.from_dict(data)
        if data.get("digest") is not None and data["digest"] != bundle.digest:
            raise SandboxSecurityError("Diff bundle digest does not match its manifest.")
        return bundle

    async def apply(self, receipt: Any, *, host_root: str | Path | None = None) -> None:
        from .apply import apply_bundle

        await asyncio.to_thread(apply_bundle, self, receipt, host_root=host_root)


def _operation_sort_key(change: FileChange) -> tuple[int, int, str]:
    if change.kind == FileChangeKind.MKDIR:
        return (0, change.path.count("/"), change.path.casefold())
    if change.kind == FileChangeKind.RMDIR:
        return (3, -change.path.count("/"), change.path.casefold())
    if change.kind == FileChangeKind.DELETE:
        return (2, -change.path.count("/"), change.path.casefold())
    return (1, change.path.count("/"), change.path.casefold())


def _make_change(
    kind: FileChangeKind,
    path: str,
    item: dict[str, Any] | None,
    current_root: str | Path | None,
    old: dict[str, Any] | None = None,
) -> FileChange:
    before_hash = None if old is None else old.get("sha256")
    if old is None and kind == FileChangeKind.DELETE:
        before_hash = None if item is None else item.get("sha256")
    before_sha256 = before_hash if isinstance(before_hash, str) else None
    if item is None:
        return FileChange(kind=kind, path=path, before_sha256=before_sha256)
    item_kind = item.get("kind")
    if kind == FileChangeKind.DELETE:
        if item_kind != "file":
            raise SandboxSecurityError(f"Only regular files can be deleted from diffs: {path!r}")
        size_value = item.get("size", 0)
        size = int(size_value) if isinstance(size_value, (int, str)) else 0
        return FileChange(
            kind=kind,
            path=path,
            before_sha256=before_sha256,
            size=size,
        )
    if item_kind != "file":
        if item_kind == "dir" and kind == FileChangeKind.ADD:
            mode_value = item.get("mode")
            mode = int(mode_value) if isinstance(mode_value, (int, str)) else None
            return FileChange(kind=FileChangeKind.MKDIR, path=path, mode=mode)
        if item_kind == "dir" and kind == FileChangeKind.RMDIR:
            return FileChange(kind=FileChangeKind.RMDIR, path=path)
        raise SandboxSecurityError(f"Only regular files and directories can be diffed: {path!r}")
    if current_root is None:
        raise SandboxSecurityError(f"Changed file has no current root: {path!r}")
    payload_item = read_file_item(current_root, path)
    data = Path(current_root).joinpath(*path.split("/")).read_bytes()
    after_hash = item.get("sha256")
    if not isinstance(after_hash, str):
        raise SandboxSecurityError(f"Changed file has no content hash: {path!r}")
    mode_value = item.get("mode")
    mode = int(mode_value) if isinstance(mode_value, (int, str)) else None
    size_value = payload_item.get("size", 0)
    size = int(size_value) if isinstance(size_value, (int, str)) else 0
    return FileChange(
        kind=kind,
        path=path,
        before_sha256=before_sha256,
        after_sha256=after_hash,
        size=size,
        mode=mode,
        payload=data,
    )
