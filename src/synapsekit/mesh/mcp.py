"""MCP tools for exposing a knowledge mesh."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..agents.base import BaseTool, ToolResult
from ..audit.types import EventKind
from ..ump.parser import UMPWriter
from ..ump.types import UMPDocument, UMPFrontmatter, UMPProvenance

if TYPE_CHECKING:
    from ..audit.redact import PIIRedactor
    from ..guardrails.policy import GuardrailPolicy
    from .core import KnowledgeMesh


@dataclass
class MemoryMCPConfig:
    """Configuration for personal memory MCP tools."""

    memory_root: Path = field(default_factory=lambda: Path.home() / ".synapsekit" / "memory")
    policy: GuardrailPolicy | None = None
    redactor: PIIRedactor | None = None

    def __post_init__(self) -> None:
        if self.policy is None:
            from ..guardrails.policy import GuardrailPolicy

            self.policy = GuardrailPolicy()
        if self.redactor is None:
            from ..audit.redact import PIIRedactor

            self.redactor = PIIRedactor()


class MeshQueryTool(BaseTool):
    """Query a ``KnowledgeMesh`` from MCP."""

    name = "mesh_query"
    description = "Query the local personal knowledge mesh with file and line citations."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Question or search query."},
            "top_k": {"type": "integer", "description": "Maximum ranked hits to return."},
        },
        "required": ["query"],
    }

    def __init__(self, mesh: KnowledgeMesh) -> None:
        self.mesh = mesh

    async def run(self, **kwargs: Any) -> ToolResult:
        top_k = kwargs.get("top_k")
        result = await self.mesh.query(
            str(kwargs.get("query", "")),
            top_k=int(top_k) if top_k is not None else None,
        )
        return ToolResult(
            output=json.dumps(
                {
                    "answer": result.answer,
                    "hits": [hit_to_dict(hit) for hit in result.hits],
                    "graph_entities": list(result.graph_entities),
                },
                indent=2,
            )
        )


class MeshReindexTool(BaseTool):
    """Reindex a ``KnowledgeMesh`` from MCP."""

    name = "mesh_reindex"
    description = "Incrementally reindex changed local mesh documents."
    parameters = {
        "type": "object",
        "properties": {
            "force": {"type": "boolean", "description": "Reindex all discovered files."}
        },
    }

    def __init__(self, mesh: KnowledgeMesh) -> None:
        self.mesh = mesh

    async def run(self, **kwargs: Any) -> ToolResult:
        summary = await self.mesh.reindex(force=bool(kwargs.get("force", False)))
        return ToolResult(output=json.dumps(summary.__dict__, indent=2))


class MeshDuplicatesTool(BaseTool):
    """Find duplicates in a ``KnowledgeMesh`` from MCP."""

    name = "mesh_duplicates"
    description = "Find likely duplicate concepts or snippets across indexed projects."
    parameters = {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "Maximum duplicate matches to return."}
        },
    }

    def __init__(self, mesh: KnowledgeMesh) -> None:
        self.mesh = mesh

    async def run(self, **kwargs: Any) -> ToolResult:
        matches = self.mesh.duplicates(limit=int(kwargs.get("limit", 20)))
        return ToolResult(output=json.dumps([match.__dict__ for match in matches], indent=2))


class MeshStatusTool(BaseTool):
    """Return ``KnowledgeMesh`` status from MCP."""

    name = "mesh_status"
    description = "Return local knowledge mesh indexing status."
    parameters = {"type": "object", "properties": {}}

    def __init__(self, mesh: KnowledgeMesh) -> None:
        self.mesh = mesh

    async def run(self, **kwargs: Any) -> ToolResult:
        return ToolResult(output=json.dumps(self.mesh.status().__dict__, indent=2))


class MemorySearchTool(BaseTool):
    """Search personal memory for ranked, redacted hits."""

    name = "memory_search"
    description = "Search personal memory for ranked, redacted hits."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "top_k": {"type": "integer", "description": "Maximum ranked hits to return."},
            "actor": {
                "type": "string",
                "description": "Identifier of the MCP client/user making the read.",
            },
        },
        "required": ["query"],
    }

    def __init__(self, mesh: KnowledgeMesh, config: MemoryMCPConfig) -> None:
        self.mesh = mesh
        self.config = config

    async def run(self, **kwargs: Any) -> ToolResult:
        query = str(kwargs.get("query", ""))
        actor = str(kwargs.get("actor", "mcp-client"))
        top_k = kwargs.get("top_k")
        result = await self.mesh.query(query, top_k=int(top_k) if top_k is not None else None)

        assert self.config.redactor is not None
        redacted_hits = [
            {**hit_to_dict(hit), "text": self.config.redactor.redact_text(hit.text)}
            for hit in result.hits
        ]

        assert self.config.policy is not None
        if self.config.policy.tracer is not None:
            self.config.policy.tracer.record(
                EventKind.MEMORY_READ,
                {"query": query, "hit_count": len(redacted_hits)},
                actor=actor,
            )

        return ToolResult(output=json.dumps({"hits": redacted_hits}, indent=2))


class MemoryRecallTool(BaseTool):
    """Recall personal memory with a synthesized answer and ranked, redacted hits."""

    name = "memory_recall"
    description = "Recall personal memory with a synthesized answer and ranked, redacted hits."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "top_k": {"type": "integer", "description": "Maximum ranked hits to return."},
            "actor": {
                "type": "string",
                "description": "Identifier of the MCP client/user making the read.",
            },
        },
        "required": ["query"],
    }

    def __init__(self, mesh: KnowledgeMesh, config: MemoryMCPConfig) -> None:
        self.mesh = mesh
        self.config = config

    async def run(self, **kwargs: Any) -> ToolResult:
        query = str(kwargs.get("query", ""))
        actor = str(kwargs.get("actor", "mcp-client"))
        top_k = kwargs.get("top_k")
        result = await self.mesh.query(query, top_k=int(top_k) if top_k is not None else None)

        assert self.config.redactor is not None
        redacted_hits = [
            {**hit_to_dict(hit), "text": self.config.redactor.redact_text(hit.text)}
            for hit in result.hits
        ]
        redacted_answer = self.config.redactor.redact_text(result.answer or "")

        assert self.config.policy is not None
        if self.config.policy.tracer is not None:
            self.config.policy.tracer.record(
                EventKind.MEMORY_READ,
                {"query": query, "hit_count": len(redacted_hits)},
                actor=actor,
            )

        return ToolResult(
            output=json.dumps(
                {"answer": redacted_answer, "hits": redacted_hits},
                indent=2,
            )
        )


class MemoryStoreTool(BaseTool):
    """Store a new personal memory item, gated by policy and recorded in the audit trail."""

    name = "memory_store"
    description = (
        "Store a new personal memory item, gated by policy and recorded in the audit trail."
    )
    parameters = {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The memory content to store."},
            "name": {"type": "string", "description": "Short human-readable name for this memory."},
            "memory_type": {
                "type": "string",
                "description": "One of: user, feedback, project, reference, general.",
            },
            "actor": {
                "type": "string",
                "description": "Identifier of the MCP client/user making the write.",
            },
        },
        "required": ["content"],
    }

    def __init__(self, mesh: KnowledgeMesh, config: MemoryMCPConfig) -> None:
        self.mesh = mesh
        self.config = config

    async def run(self, **kwargs: Any) -> ToolResult:
        content = str(kwargs.get("content", ""))
        actor = str(kwargs.get("actor", "mcp-client"))

        assert self.config.policy is not None
        report = await self.config.policy.check_input(content)
        if not report.allowed:
            if self.config.policy.tracer is not None:
                self.config.policy.tracer.record(
                    EventKind.MEMORY_WRITE,
                    {
                        "blocked": True,
                        "reasons": [f"{f.guard}: {f.detail}" for f in report.triggered],
                    },
                    actor=actor,
                )
            return ToolResult(
                output="",
                error=f"memory_store blocked by policy: {[f.detail for f in report.findings]}",
            )

        name = str(kwargs.get("name", "")) or _default_name(content)
        memory_type = kwargs.get("memory_type", "general")

        doc = UMPDocument(
            frontmatter=UMPFrontmatter(
                name=name,
                type=memory_type,  # type: ignore[arg-type]
                scope="global",
                visibility="local",
                provenance=UMPProvenance(authors=[actor]),
            ),
            body=content,
        )
        memory_root = Path(self.config.memory_root).expanduser()
        path = memory_root / f"{_slugify(doc.frontmatter.name)}-{_new_id()}.md"
        await UMPWriter.write_file(doc, path)

        if self.config.policy.tracer is not None:
            self.config.policy.tracer.record(
                EventKind.MEMORY_WRITE,
                {"path": str(path), "name": doc.frontmatter.name, "actor": actor},
                actor=actor,
            )

        await self.mesh.reindex(force=False)
        return ToolResult(output=json.dumps({"stored": True, "path": str(path)}))


def build_mesh_tools(
    mesh: KnowledgeMesh, memory_config: MemoryMCPConfig | None = None
) -> list[BaseTool]:
    """Return all MCP tools for ``mesh``."""

    if memory_config is None:
        mesh_roots = mesh.config.expanded_roots()
        default_root = Path.home() / ".synapsekit" / "memory"
        if default_root not in mesh_roots and mesh_roots:
            memory_root = mesh_roots[0]
        else:
            memory_root = default_root
        config = MemoryMCPConfig(memory_root=memory_root)
    else:
        config = memory_config

    _validate_memory_root(mesh, config)

    return [
        MeshQueryTool(mesh),
        MeshReindexTool(mesh),
        MeshDuplicatesTool(mesh),
        MeshStatusTool(mesh),
        MemorySearchTool(mesh, config),
        MemoryRecallTool(mesh, config),
        MemoryStoreTool(mesh, config),
    ]


def _validate_memory_root(mesh: KnowledgeMesh, config: MemoryMCPConfig) -> None:
    expanded_memory_root = Path(config.memory_root).expanduser()
    mesh_roots = mesh.config.expanded_roots()
    if expanded_memory_root not in mesh_roots:
        raise ValueError(
            f"memory_root '{expanded_memory_root}' is not one of this mesh's configured roots ({mesh_roots}); "
            f"add it to MeshConfig(roots=[...]) or pass a memory_root that already is one."
        )


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^\w]+", "-", text.lower()).strip("-")
    return cleaned or "memory"


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def _default_name(content: str) -> str:
    first_line = content.strip().splitlines()[0] if content.strip() else "memory"
    return first_line[:30].strip() or "memory"


def hit_to_dict(hit: Any) -> dict[str, Any]:
    """Serialize a mesh hit without requiring callers to import dataclasses."""

    return {
        "text": hit.text,
        "score": hit.score,
        "path": hit.path,
        "line_start": hit.line_start,
        "line_end": hit.line_end,
        "headings": list(hit.headings),
        "repo_root": hit.repo_root,
        "metadata": hit.metadata,
    }
