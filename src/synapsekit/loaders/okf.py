"""Open Knowledge Format (OKF v0.1) loader.

OKF is a vendor-neutral, low-tech format for the curated knowledge agents need:
a directory tree of Markdown files, one per *concept* (a dataset, table, metric,
...). Each file is YAML frontmatter (``type`` required; ``title``,
``description``, ``resource``, ``tags``, ``timestamp`` optional) plus a Markdown
body, and concepts cross-link via ordinary relative Markdown links — so a bundle
is effectively a knowledge graph on disk.

This loader walks a bundle and yields one :class:`Document` per concept, with the
cross-links resolved into ``metadata["linked_concepts"]`` (bundle-relative
paths). That resolved link structure is what the #825 ``okf_to_world_model``
adapter consumes to build a graph without any lossy LLM/heuristic extraction.

Spec + reference bundles: https://github.com/GoogleCloudPlatform/knowledge-catalog/tree/main/okf
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any

from ..mesh.loaders import _FRONTMATTER_RE
from .base import Document

logger = logging.getLogger(__name__)

# Markdown inline link: [label](target)
_MD_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
# A URI scheme prefix (http:, https:, mailto:, ...) — an external, non-bundle link.
_SCHEME_RE = re.compile(r"\A[a-z][a-z0-9+.\-]*:", re.IGNORECASE)

# Reserved OKF filenames treated as navigation stubs, skipped unless opted in.
_RESERVED_STUBS = frozenset({"index.md"})
# OKF standard optional frontmatter fields lifted to the top of ``metadata``.
_PASSTHROUGH_FIELDS = ("title", "description", "resource", "tags", "timestamp")


def _require_yaml() -> Any:
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for OpenKnowledgeFormatLoader. "
            "Install it with `pip install synapsekit[okf]`."
        ) from None
    return yaml


class OpenKnowledgeFormatLoader:
    """Load an Open Knowledge Format bundle into ``Document`` objects.

    Args:
        path: Root of the OKF bundle (a directory) or a single ``.md`` concept.
        recursive: Walk subdirectories (only relevant for a directory ``path``).
        resolve_links: Parse Markdown cross-links into ``metadata["linked_concepts"]``.
        include_index: Include reserved navigation stubs (``index.md``).
        require_type: Skip concept files with no OKF ``type`` frontmatter
            (spec conformance). When ``False`` they are emitted with
            ``okf_type=None`` instead.
        encoding: Text encoding used to read concept files.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        recursive: bool = True,
        resolve_links: bool = True,
        include_index: bool = False,
        require_type: bool = True,
        encoding: str = "utf-8",
    ) -> None:
        self.path = Path(path).expanduser()
        self.recursive = recursive
        self.resolve_links = resolve_links
        self.include_index = include_index
        self.require_type = require_type
        self.encoding = encoding

    def load(self) -> list[Document]:
        _require_yaml()  # fail fast with the install hint before touching the disk
        bundle_root, files = self._discover()
        docs: list[Document] = []
        for file in files:
            doc = self._load_concept(file, bundle_root)
            if doc is not None:
                docs.append(doc)
        return docs

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)

    # ------------------------------------------------------------------ #
    # Discovery
    # ------------------------------------------------------------------ #

    def _discover(self) -> tuple[Path, list[Path]]:
        """Return ``(bundle_root, concept_files)`` in deterministic order."""
        if self.path.is_file():
            return self.path.parent, [self.path]
        if not self.path.is_dir():
            return self.path, []

        files: list[Path] = []
        if self.recursive:
            for current_root, dirs, names in os.walk(self.path):
                dirs.sort()
                for name in names:
                    if self._is_concept_file(name):
                        files.append(Path(current_root) / name)
        else:
            for entry in self.path.iterdir():
                if entry.is_file() and self._is_concept_file(entry.name):
                    files.append(entry)

        files.sort(key=lambda p: self._relative(p, self.path))
        return self.path, files

    def _is_concept_file(self, name: str) -> bool:
        if not name.lower().endswith(".md"):
            return False
        return self.include_index or name.lower() not in _RESERVED_STUBS

    # ------------------------------------------------------------------ #
    # Per-concept parsing
    # ------------------------------------------------------------------ #

    def _load_concept(self, file: Path, bundle_root: Path) -> Document | None:
        concept_path = self._relative(file, bundle_root)
        try:
            raw = file.read_text(encoding=self.encoding, errors="replace")
        except OSError as exc:
            logger.warning("OKF: could not read concept %s: %s", concept_path, exc)
            return None

        frontmatter, body = self._parse(raw, concept_path)
        okf_type = frontmatter.get("type")
        if okf_type is None and self.require_type:
            logger.warning(
                "OKF: skipping %s — no `type` frontmatter (require_type=True).", concept_path
            )
            return None

        metadata: dict[str, Any] = {
            "source": "okf",
            "okf_type": okf_type,
            "concept_path": concept_path,
            "bundle_root": str(bundle_root.resolve()),
            "frontmatter": frontmatter,
        }
        for field in _PASSTHROUGH_FIELDS:
            if field in frontmatter:
                metadata[field] = frontmatter[field]
        if self.resolve_links:
            metadata["linked_concepts"] = self._resolve_links(body, file, bundle_root)

        return Document(text=body, metadata=metadata)

    def _parse(self, raw: str, concept_path: str) -> tuple[dict[str, Any], str]:
        """Split YAML frontmatter from the Markdown body.

        Reuses the mesh loader's frontmatter regex for detection/stripping.
        Malformed YAML surfaces a warning and is treated as no frontmatter,
        never a crash.
        """
        match = _FRONTMATTER_RE.match(raw)
        if match is None:
            return {}, raw.strip()

        body = raw[match.end() :].strip()
        block_lines = match.group(0).splitlines()[1:]  # drop opening ---
        while block_lines and block_lines[-1].strip() == "---":
            block_lines.pop()  # drop closing ---
        payload = "\n".join(block_lines)

        yaml = _require_yaml()
        try:
            parsed = yaml.safe_load(payload)
        except yaml.YAMLError as exc:
            logger.warning("OKF: malformed YAML frontmatter in %s: %s", concept_path, exc)
            return {}, body
        if not isinstance(parsed, dict):
            return {}, body
        return parsed, body

    def _resolve_links(self, body: str, file: Path, bundle_root: Path) -> list[str]:
        """Resolve in-bundle Markdown cross-links to bundle-relative concept paths.

        External links (schemes, protocol-relative), pure anchors, and links that
        don't resolve to an existing ``.md`` file inside the bundle are excluded;
        they remain in the body text but are not treated as graph edges.
        """
        root = bundle_root.resolve()
        resolved: set[str] = set()
        for target in _MD_LINK_RE.findall(body):
            target = target.strip()
            # Markdown allows an optional title: [x](path "title") — drop it.
            if " " in target:
                target = target.split(" ", 1)[0]
            target = target.split("#", 1)[0].split("?", 1)[0]
            if not target or target.startswith("//") or _SCHEME_RE.match(target):
                continue
            if target.startswith("/"):
                candidate = root / target.lstrip("/")
            else:
                candidate = file.parent / target
            try:
                candidate = candidate.resolve()
            except OSError:
                continue
            if candidate.suffix.lower() != ".md" or not candidate.is_file():
                continue
            if not self._within(candidate, root):
                continue
            resolved.add(self._relative(candidate, root))
        return sorted(resolved)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _relative(path: Path, root: Path) -> str:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            return path.name

    @staticmethod
    def _within(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False


# Short alias, per the issue.
OKFLoader = OpenKnowledgeFormatLoader

__all__ = ["OKFLoader", "OpenKnowledgeFormatLoader"]
