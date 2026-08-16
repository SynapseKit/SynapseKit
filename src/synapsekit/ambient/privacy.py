"""Per-source opt-out for the ambient daemon.

Both sources are enabled by default; listing a source name in the ignore
file disables it. Same tiny parsing shape as
``synapsekit.mesh.privacy.MeshPrivacyFilter._read_ignore_file``, applied to
source names instead of path globs.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_AMBIENT_IGNORE = Path.home() / ".synapsekit" / "ambient.ignore"


def load_disabled_sources(path: str | Path | None = DEFAULT_AMBIENT_IGNORE) -> set[str]:
    if path is None:
        return set()
    ignore_path = Path(path).expanduser()
    if not ignore_path.exists():
        return set()

    disabled: set[str] = set()
    for raw in ignore_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        disabled.add(line)
    return disabled
