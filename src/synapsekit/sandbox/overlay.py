"""Secure, deterministic filesystem snapshots and a materialized overlay.

Native CoW is backend-specific.  The portable fallback used by the core is a
materialized working tree with a manifest; backends report whether they can
replace it with a native snapshot.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import shutil
import stat
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .errors import SandboxSecurityError
from .types import SnapshotManifest, normalize_relative_path


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest_fingerprint(items: Iterable[dict[str, object]]) -> str:
    encoded = json.dumps(
        list(items),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _digest(encoded)


def _matches(path: str, patterns: tuple[str, ...]) -> bool:
    normalized = path.strip("/")
    for raw in patterns:
        pattern = raw.replace("\\", "/").strip("/")
        if not pattern:
            continue
        if normalized == pattern or normalized.startswith(pattern + "/"):
            return True
        if fnmatch.fnmatch(normalized, pattern):
            return True
    return False


def _could_contain_include(path: str, includes: tuple[str, ...]) -> bool:
    if not includes:
        return True
    normalized = path.strip("/")
    return any(
        normalized == pattern.strip("/")
        or pattern.strip("/").startswith(normalized + "/")
        or fnmatch.fnmatch(normalized, pattern.strip("/"))
        for pattern in includes
    )


def _ensure_inside(root: Path, candidate: Path) -> None:
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise SandboxSecurityError(f"Symlink escapes sandbox base: {resolved!s}") from exc


def _entry_item(root: Path, path: Path, relative: str) -> dict[str, object]:
    info = path.lstat()
    mode = stat.S_IMODE(info.st_mode)
    if stat.S_ISLNK(info.st_mode):
        target = os.readlink(path)
        _ensure_inside(root, path.parent / target)
        return {"kind": "symlink", "path": relative, "target": target, "mode": mode}
    if stat.S_ISREG(info.st_mode):
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise SandboxSecurityError(f"Cannot read snapshot file {relative!r}.") from exc
        return {
            "kind": "file",
            "path": relative,
            "sha256": _digest(data),
            "size": len(data),
            "mode": mode,
        }
    if stat.S_ISDIR(info.st_mode):
        return {"kind": "dir", "path": relative, "mode": mode}
    raise SandboxSecurityError(f"Unsupported filesystem entry in snapshot: {relative!r}")


def capture_manifest(
    root: str | Path,
    *,
    include: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
) -> SnapshotManifest:
    """Capture a deterministic manifest without following symlinks."""

    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"Sandbox base directory does not exist: {base}")

    items: list[dict[str, object]] = []

    def walk(directory: Path, parent: str = "") -> None:
        try:
            entries = sorted(directory.iterdir(), key=lambda item: item.name.casefold())
        except OSError as exc:
            raise SandboxSecurityError(f"Cannot list snapshot directory {directory}.") from exc

        for path in entries:
            relative = normalize_relative_path(f"{parent}/{path.name}" if parent else path.name)
            if _matches(relative, exclude):
                continue
            if (
                include
                and not _matches(relative, include)
                and not _could_contain_include(relative, include)
            ):
                continue
            item = _entry_item(base, path, relative)
            if item["kind"] == "dir":
                items.append(item)
                walk(path, relative)
            elif not include or _matches(relative, include):
                items.append(item)

    walk(base)
    items.sort(key=lambda item: str(item["path"]).casefold())
    return SnapshotManifest(
        root=str(base),
        items=tuple(items),
        fingerprint=_manifest_fingerprint(items),
    )


def _copy_entry(source: Path, destination: Path, root: Path) -> None:
    info = source.lstat()
    if stat.S_ISLNK(info.st_mode):
        target = os.readlink(source)
        _ensure_inside(root, source.parent / target)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(target, target_is_directory=source.is_dir())
        return
    if stat.S_ISDIR(info.st_mode):
        destination.mkdir(parents=True, exist_ok=True)
        for child in source.iterdir():
            _copy_entry(child, destination / child.name, root)
        shutil.copystat(source, destination, follow_symlinks=False)
        return
    if stat.S_ISREG(info.st_mode):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination, follow_symlinks=False)
        return
    raise SandboxSecurityError(f"Unsupported filesystem entry: {source}")


def clone_tree(
    source: str | Path,
    destination: str | Path,
    *,
    include: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
) -> SnapshotManifest:
    """Materialize a safe selected copy and return its source manifest."""

    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    if not source_path.is_dir():
        raise FileNotFoundError(f"Sandbox base directory does not exist: {source_path}")
    if destination_path.exists():
        raise FileExistsError(f"Sandbox overlay destination already exists: {destination_path}")

    manifest = capture_manifest(source_path, include=include, exclude=exclude)
    destination_path.mkdir(parents=True, exist_ok=False)
    selected = {str(item["path"]): item for item in manifest.items}

    for relative, item in sorted(selected.items()):
        target = destination_path.joinpath(*relative.split("/"))
        source_entry = source_path.joinpath(*relative.split("/"))
        if item["kind"] == "dir":
            target.mkdir(parents=True, exist_ok=True)
        elif item["kind"] == "file" or item["kind"] == "symlink":
            _copy_entry(source_entry, target, source_path)

    return manifest


@dataclass(frozen=True, slots=True)
class FsOverlay:
    """Portable filesystem overlay contract used by a sandbox session.

    This implementation materializes a selected copy. A backend that offers a
    native CoW snapshot can wrap the same contract and set ``native_cow`` on
    its capability report without changing diff or apply semantics.
    """

    source: str | Path
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", Path(self.source).expanduser().resolve())

    @property
    def native_cow(self) -> bool:
        """Whether this portable overlay provides native copy-on-write."""

        return False

    def snapshot(self) -> SnapshotManifest:
        """Capture the selected source tree without materializing it."""

        return capture_manifest(self.source, include=self.include, exclude=self.exclude)

    def materialize(self, destination: str | Path) -> SnapshotManifest:
        """Create the isolated working tree and return its baseline manifest."""

        return clone_tree(
            self.source,
            destination,
            include=self.include,
            exclude=self.exclude,
        )


def read_file_item(root: str | Path, relative: str) -> dict[str, object]:
    """Read one current overlay entry into the manifest item shape."""

    base = Path(root).resolve()
    safe = normalize_relative_path(relative)
    path = base.joinpath(*safe.split("/"))
    try:
        path.resolve(strict=False).relative_to(base)
    except ValueError as exc:
        raise SandboxSecurityError(f"Path escapes sandbox root: {relative!r}") from exc
    return _entry_item(base, path, safe)
