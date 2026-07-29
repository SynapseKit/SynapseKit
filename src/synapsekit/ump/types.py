from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

UMPType = Literal["user", "feedback", "project", "reference", "general"]
UMPScope = Literal["global", "project", "session"]
UMPVisibility = Literal["local", "shared", "team"]


@dataclass
class UMPProvenance:
    authors: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    signed_by: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UMPProvenance:
        return cls(
            authors=data.get("authors", []),
            evidence=data.get("evidence", []),
            signed_by=data.get("signed_by", ""),
        )


@dataclass
class UMPFrontmatter:
    ump_version: str = "1.0"
    name: str = ""
    type: UMPType = "general"
    scope: UMPScope = "project"
    visibility: UMPVisibility = "local"
    provenance: UMPProvenance = field(default_factory=UMPProvenance)
    links: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        res = asdict(self)
        res["provenance"] = self.provenance.to_dict()
        return res

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UMPFrontmatter:
        prov_data = data.get("provenance", {})
        prov = (
            UMPProvenance.from_dict(prov_data) if isinstance(prov_data, dict) else UMPProvenance()
        )
        return cls(
            ump_version=data.get("ump_version", "1.0"),
            name=data.get("name", ""),
            type=data.get("type", "general"),
            scope=data.get("scope", "project"),
            visibility=data.get("visibility", "local"),
            provenance=prov,
            links=data.get("links", []),
        )


@dataclass
class UMPDocument:
    frontmatter: UMPFrontmatter = field(default_factory=UMPFrontmatter)
    body: str = ""
    source_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "frontmatter": self.frontmatter.to_dict(),
            "body": self.body,
            "source_path": self.source_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UMPDocument:
        fm_data = data.get("frontmatter", {})
        fm = UMPFrontmatter.from_dict(fm_data) if isinstance(fm_data, dict) else UMPFrontmatter()
        return cls(
            frontmatter=fm,
            body=data.get("body", ""),
            source_path=data.get("source_path", ""),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
