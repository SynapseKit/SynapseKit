from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import ClassVar

import yaml

from synapsekit.ump.types import UMPDocument, UMPFrontmatter, UMPProvenance


class BaseUMPAdapter:
    """Base adapter class for converting external tool memory/rule files to UMP."""

    tool_name: ClassVar[str] = "generic"
    default_filename: ClassVar[str] = ""

    @classmethod
    def detect(cls, base_dir: str | Path) -> bool:
        if not cls.default_filename:
            return False
        path = Path(base_dir) / cls.default_filename
        return path.exists()

    @classmethod
    async def to_ump(cls, path: str | Path) -> UMPDocument:
        raise NotImplementedError

    @classmethod
    def from_ump(cls, doc: UMPDocument) -> str:
        return doc.body


class ClaudeAdapter(BaseUMPAdapter):
    """Adapter for Claude Code memory format (CLAUDE.md)."""

    tool_name = "claude-code"
    default_filename = "CLAUDE.md"

    @classmethod
    async def to_ump(cls, path: str | Path) -> UMPDocument:
        file_path = Path(path)
        content = (
            await asyncio.to_thread(file_path.read_text, encoding="utf-8")
            if file_path.exists()
            else ""
        )
        fm = UMPFrontmatter(
            name="claude-memory",
            type="project",
            scope="project",
            visibility="local",
            provenance=UMPProvenance(authors=["agent:claude-code"]),
        )
        return UMPDocument(frontmatter=fm, body=content, source_path=str(file_path.resolve()))


class CursorAdapter(BaseUMPAdapter):
    """Adapter for Cursor rules format (.cursor/rules or .cursorrules)."""

    tool_name = "cursor"
    default_filename = ".cursorrules"

    @classmethod
    def detect(cls, base_dir: str | Path) -> bool:
        p1 = Path(base_dir) / ".cursorrules"
        p2 = Path(base_dir) / ".cursor" / "rules"
        return p1.exists() or p2.exists()

    @classmethod
    async def to_ump(cls, path: str | Path) -> UMPDocument:
        file_path = Path(path)
        content = (
            await asyncio.to_thread(file_path.read_text, encoding="utf-8")
            if file_path.exists()
            else ""
        )
        fm = UMPFrontmatter(
            name="cursor-rules",
            type="project",
            scope="project",
            visibility="local",
            provenance=UMPProvenance(authors=["agent:cursor"]),
        )
        return UMPDocument(frontmatter=fm, body=content, source_path=str(file_path.resolve()))


class AiderAdapter(BaseUMPAdapter):
    """Adapter for Aider configuration (.aider.conf.yml)."""

    tool_name = "aider"
    default_filename = ".aider.conf.yml"

    @classmethod
    async def to_ump(cls, path: str | Path) -> UMPDocument:
        file_path = Path(path)
        content = ""
        if file_path.exists():
            raw_text = await asyncio.to_thread(file_path.read_text, encoding="utf-8")
            try:
                data = yaml.safe_load(raw_text)
                content = json.dumps(data, indent=2) if data else raw_text
            except Exception:
                content = raw_text

        fm = UMPFrontmatter(
            name="aider-config",
            type="project",
            scope="project",
            visibility="local",
            provenance=UMPProvenance(authors=["agent:aider"]),
        )
        return UMPDocument(frontmatter=fm, body=content, source_path=str(file_path.resolve()))


class ContinueAdapter(BaseUMPAdapter):
    """Adapter for Continue dev configuration (.continue/config.json)."""

    tool_name = "continue"
    default_filename = ".continue/config.json"

    @classmethod
    async def to_ump(cls, path: str | Path) -> UMPDocument:
        file_path = Path(path)
        content = ""
        if file_path.exists():
            raw_text = await asyncio.to_thread(file_path.read_text, encoding="utf-8")
            try:
                data = json.loads(raw_text)
                content = json.dumps(data, indent=2)
            except Exception:
                content = raw_text

        fm = UMPFrontmatter(
            name="continue-config",
            type="project",
            scope="project",
            visibility="local",
            provenance=UMPProvenance(authors=["agent:continue"]),
        )
        return UMPDocument(frontmatter=fm, body=content, source_path=str(file_path.resolve()))


async def auto_detect_and_convert(directory: str | Path) -> list[UMPDocument]:
    """Scan directory for known tool memory files and convert to UMP documents."""
    base = Path(directory)
    results: list[UMPDocument] = []

    if ClaudeAdapter.detect(base):
        results.append(await ClaudeAdapter.to_ump(base / ClaudeAdapter.default_filename))

    if CursorAdapter.detect(base):
        cursor_path = (
            base / ".cursorrules"
            if (base / ".cursorrules").exists()
            else base / ".cursor" / "rules"
        )
        results.append(await CursorAdapter.to_ump(cursor_path))

    if AiderAdapter.detect(base):
        results.append(await AiderAdapter.to_ump(base / AiderAdapter.default_filename))

    if ContinueAdapter.detect(base):
        results.append(await ContinueAdapter.to_ump(base / ContinueAdapter.default_filename))

    return results
