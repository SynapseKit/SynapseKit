"""Preflighted, rollback-based host application of diff bundles."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any

from .diff import DiffBundle
from .errors import ApplyConflictError, SandboxSecurityError
from .types import FileChange, FileChangeKind, normalize_relative_path


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _safe_target(root: Path, relative: str) -> Path:
    safe = normalize_relative_path(relative)
    target = root.joinpath(*safe.split("/"))
    current = root
    for component in safe.split("/")[:-1]:
        current = current / component
        if current.is_symlink():
            raise SandboxSecurityError(f"Diff target traverses a symlink: {relative!r}")
    try:
        target.parent.resolve(strict=False).relative_to(root.resolve())
    except ValueError as exc:
        raise SandboxSecurityError(f"Diff target escapes host root: {relative!r}") from exc
    return target


def _matches_expected(path: Path, change: FileChange) -> bool:
    if change.kind in {FileChangeKind.ADD, FileChangeKind.MKDIR}:
        return not _exists(path)
    if not _exists(path):
        return False
    if change.before_sha256 is not None:
        return (
            path.is_file() and not path.is_symlink() and _file_digest(path) == change.before_sha256
        )
    if change.kind == FileChangeKind.DELETE:
        return path.is_dir() and not path.is_symlink()
    return True


def _backup(path: Path, backup: Path) -> None:
    if path.is_symlink():
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.symlink_to(os.readlink(path), target_is_directory=path.is_dir())
    elif path.is_dir():
        shutil.copytree(path, backup)
    else:
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _apply_change(root: Path, change: FileChange) -> None:
    target = _safe_target(root, change.path)
    if change.kind == FileChangeKind.MKDIR:
        target.mkdir(parents=True, exist_ok=False)
        if change.mode is not None:
            target.chmod(stat.S_IMODE(change.mode))
        return
    if change.kind in {FileChangeKind.DELETE, FileChangeKind.RMDIR}:
        _remove(target)
        return
    if change.payload is None:
        raise SandboxSecurityError(f"File operation is missing payload: {change.path!r}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.synapsekit-tmp")
    temporary.write_bytes(change.payload)
    if change.mode is not None:
        temporary.chmod(stat.S_IMODE(change.mode))
    os.replace(temporary, target)


def apply_bundle(
    bundle: DiffBundle,
    receipt: Any,
    *,
    host_root: str | Path | None = None,
) -> None:
    """Apply a bundle transactionally, restoring the host on failure."""

    if not getattr(receipt, "passed", False):
        raise ApplyConflictError("A diff cannot be applied without a passing evaluation receipt.")
    if getattr(receipt, "diff_sha256", None) != bundle.digest:
        raise ApplyConflictError("Evaluation receipt does not match this diff bundle.")

    root = Path(host_root or bundle.host_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Host root does not exist: {root}")

    targets = [(_safe_target(root, change.path), change) for change in bundle.changes]
    for target, change in targets:
        if not _matches_expected(target, change):
            raise ApplyConflictError(f"Host changed since snapshot: {change.path}")

    journal = Path(tempfile.mkdtemp(prefix=".synapsekit-apply-", dir=str(root.parent)))
    backups: list[tuple[Path, Path]] = []
    created: list[Path] = []
    try:
        for index, (target, _change) in enumerate(targets):
            if _exists(target):
                backup = journal / str(index)
                _backup(target, backup)
                backups.append((target, backup))

        for target, change in targets:
            if change.kind in {FileChangeKind.ADD, FileChangeKind.MKDIR}:
                created.append(target)
            _apply_change(root, change)

        for target, change in targets:
            if change.after_sha256 is not None and (
                not target.is_file() or _file_digest(target) != change.after_sha256
            ):
                raise ApplyConflictError(f"Post-apply verification failed: {change.path}")
    except Exception:
        for target in sorted(created, key=lambda item: len(item.parts), reverse=True):
            if _exists(target):
                _remove(target)
        for target, backup in reversed(backups):
            if _exists(target):
                _remove(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            if backup.is_dir() and not backup.is_symlink():
                shutil.copytree(backup, target)
            elif backup.is_symlink():
                target.symlink_to(os.readlink(backup), target_is_directory=backup.is_dir())
            else:
                shutil.copy2(backup, target)
        raise
    finally:
        shutil.rmtree(journal, ignore_errors=True)
