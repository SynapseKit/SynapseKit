from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from synapsekit.ump.types import UMPDocument, UMPFrontmatter

WIKILINK_PATTERN = re.compile(r"\[\[([^\]]+)\]\]")


class UMPReader:
    """Parse UMP-formatted markdown files into UMPDocument objects."""

    @classmethod
    def parse(cls, content: str, *, source_path: str = "") -> UMPDocument:
        frontmatter_dict, body = cls._parse_yaml_frontmatter(content)
        frontmatter = UMPFrontmatter.from_dict(frontmatter_dict)

        # Automatically discover wikilinks in body if links not specified in frontmatter
        extracted_links = cls._extract_wikilinks(body)
        if extracted_links and not frontmatter.links:
            frontmatter.links = extracted_links
        elif extracted_links:
            # Merge extracted wikilinks into existing links without duplicates
            for link in extracted_links:
                if link not in frontmatter.links:
                    frontmatter.links.append(link)

        return UMPDocument(
            frontmatter=frontmatter,
            body=body,
            source_path=source_path,
        )

    @classmethod
    def read_file(cls, path: str | Path) -> UMPDocument:
        file_path = Path(path)
        content = file_path.read_text(encoding="utf-8")
        return cls.parse(content, source_path=str(file_path.resolve()))

    @classmethod
    def _parse_yaml_frontmatter(cls, raw: str) -> tuple[dict[str, Any], str]:
        if not raw.startswith("---"):
            return {}, raw.strip()

        parts = raw.split("---", 2)
        if len(parts) < 3:
            return {}, raw.strip()

        yaml_str = parts[1].strip()
        body = parts[2].strip()

        try:
            parsed = yaml.safe_load(yaml_str)
            if isinstance(parsed, dict):
                return parsed, body
        except Exception:
            pass

        return {}, body

    @classmethod
    def _extract_wikilinks(cls, body: str) -> list[str]:
        matches = WIKILINK_PATTERN.findall(body)
        seen: set[str] = set()
        result: list[str] = []
        for match in matches:
            cleaned = f"[[{match.strip()}]]"
            if cleaned not in seen:
                seen.add(cleaned)
                result.append(cleaned)
        return result


class UMPWriter:
    """Serialize UMPDocument objects to UMP-formatted markdown."""

    @classmethod
    def serialize(cls, doc: UMPDocument) -> str:
        yaml_str = cls._render_yaml_frontmatter(doc.frontmatter)
        return f"---\n{yaml_str}---\n\n{doc.body}\n"

    @classmethod
    def write_file(cls, doc: UMPDocument, path: str | Path) -> None:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        content = cls.serialize(doc)
        file_path.write_text(content, encoding="utf-8")

    @classmethod
    def _render_yaml_frontmatter(cls, fm: UMPFrontmatter) -> str:
        data = fm.to_dict()
        # Clean up empty values for concise YAML
        if not data["provenance"]["authors"]:
            del data["provenance"]["authors"]
        if not data["provenance"]["evidence"]:
            del data["provenance"]["evidence"]
        if not data["provenance"]["signed_by"]:
            del data["provenance"]["signed_by"]

        yaml_output = yaml.safe_dump(
            data,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )
        return str(yaml_output)
