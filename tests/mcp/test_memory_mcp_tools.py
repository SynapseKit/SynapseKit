from __future__ import annotations

import asyncio
import base64
import json
import sys
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from synapsekit.audit import verify
from synapsekit.audit.types import EventKind, Verdict
from synapsekit.guardrails.input_guards import TopicGuard
from synapsekit.guardrails.policy import GuardrailPolicy
from synapsekit.mcp.server.core import MCPServer
from synapsekit.mesh import KnowledgeMesh, MeshConfig, build_mesh_tools
from synapsekit.mesh.mcp import (
    MemoryMCPConfig,
    MemoryRecallTool,
    MemorySearchTool,
    MemoryStoreTool,
)
from synapsekit.ump.parser import UMPReader


def _manifest_keys_as_trusted(path: str | Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    keys = manifest.get("keys") or manifest.get("original_manifest", {}).get("keys", {})
    return {key_id: base64.b64decode(info["public_key_b64"]) for key_id, info in keys.items()}


def test_memory_store_writes_ump_file_and_becomes_searchable(tmp_path: Path) -> None:
    async def run() -> None:
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        mesh = KnowledgeMesh(
            MeshConfig(
                roots=[memory_dir],
                state_dir=tmp_path / "state",
                vector_backend="memory",
                graph_backend="memory",
                use_git=False,
            )
        )

        config = MemoryMCPConfig(memory_root=memory_dir)
        store_tool = MemoryStoreTool(mesh, config)
        search_tool = MemorySearchTool(mesh, config)

        res = await store_tool.run(
            content="Important preference: user prefers dark mode UI.",
            name="UI Preference",
            memory_type="user",
            actor="test-agent",
        )
        data = json.loads(res.output)
        assert data["stored"] is True
        written_path = Path(data["path"])
        assert written_path.exists()
        assert written_path.parent == memory_dir

        # Validate UMP document structure
        doc = await UMPReader.read_file(written_path)
        assert doc.frontmatter.name == "UI Preference"
        assert doc.frontmatter.type == "user"
        assert doc.frontmatter.provenance.authors == ["test-agent"]
        assert doc.body == "Important preference: user prefers dark mode UI."

        # Verify searchability
        search_res = await search_tool.run(query="dark mode")
        search_data = json.loads(search_res.output)
        assert len(search_data["hits"]) > 0
        assert "dark mode" in search_data["hits"][0]["text"]

    asyncio.run(run())


def test_memory_store_blocked_by_policy_writes_nothing(tmp_path: Path) -> None:
    async def run() -> None:
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        mesh = KnowledgeMesh(
            MeshConfig(
                roots=[memory_dir],
                state_dir=tmp_path / "state",
                vector_backend="memory",
                graph_backend="memory",
                use_git=False,
            )
        )

        # Policy with guard blocking "confidential"
        policy = GuardrailPolicy(guards=[TopicGuard(blocked_topics=["confidential"])])
        config = MemoryMCPConfig(memory_root=memory_dir, policy=policy)
        store_tool = MemoryStoreTool(mesh, config)

        res = await store_tool.run(
            content="This is confidential secret key material.",
            name="Confidential Data",
            actor="custom-writer",
        )
        assert res.is_error is True
        assert "memory_store blocked by policy" in res.error

        # Assert no files were created
        assert list(memory_dir.glob("*.md")) == []

        # Assert audit recorded the blocked attempt attributed to the caller actor
        assert policy.tracer is not None
        records = policy.tracer.records
        blocked_rec = next(
            r
            for r in records
            if r.kind == EventKind.MEMORY_WRITE and r.payload.get("blocked") is True
        )
        assert blocked_rec.actor == "custom-writer"
        assert blocked_rec.payload["blocked"] is True
        # Assert raw blocked memory body content does not leak into audit payload
        from synapsekit.audit.types import deep_unfreeze

        assert "secret key material" not in json.dumps(deep_unfreeze(blocked_rec.payload))

    asyncio.run(run())


def test_memory_search_redacts_pii_in_hits(tmp_path: Path) -> None:
    async def run() -> None:
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        mesh = KnowledgeMesh(
            MeshConfig(
                roots=[memory_dir],
                state_dir=tmp_path / "state",
                vector_backend="memory",
                graph_backend="memory",
                use_git=False,
            )
        )
        config = MemoryMCPConfig(memory_root=memory_dir)

        store_tool = MemoryStoreTool(mesh, config)
        await store_tool.run(
            content="Support email is alice@example.com for inquiries.",
            name="Support Contact",
        )

        search_tool = MemorySearchTool(mesh, config)
        search_res = await search_tool.run(query="Support email")
        search_data = json.loads(search_res.output)

        assert len(search_data["hits"]) > 0
        hit_text = search_data["hits"][0]["text"]
        assert "alice@example.com" not in hit_text
        assert "[REDACTED:EMAIL]" in hit_text

    asyncio.run(run())


def test_memory_recall_redacts_answer(tmp_path: Path) -> None:
    async def run() -> None:
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        mesh = KnowledgeMesh(
            MeshConfig(
                roots=[memory_dir],
                state_dir=tmp_path / "state",
                vector_backend="memory",
                graph_backend="memory",
                use_git=False,
            )
        )
        config = MemoryMCPConfig(memory_root=memory_dir)

        store_tool = MemoryStoreTool(mesh, config)
        await store_tool.run(
            content="Owner contact SSN is 123-45-6789.",
            name="Owner SSN",
        )

        recall_tool = MemoryRecallTool(mesh, config)
        recall_res = await recall_tool.run(query="Owner contact SSN")
        recall_data = json.loads(recall_res.output)

        assert "123-45-6789" not in recall_data["answer"]
        assert "hits" in recall_data
        assert len(recall_data["hits"]) > 0
        assert "123-45-6789" not in recall_data["hits"][0]["text"]

    asyncio.run(run())


def test_memory_read_and_write_emit_audit_records(tmp_path: Path) -> None:
    async def run() -> None:
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        mesh = KnowledgeMesh(
            MeshConfig(
                roots=[memory_dir],
                state_dir=tmp_path / "state",
                vector_backend="memory",
                graph_backend="memory",
                use_git=False,
            )
        )

        policy = GuardrailPolicy()
        config = MemoryMCPConfig(memory_root=memory_dir, policy=policy)

        store_tool = MemoryStoreTool(mesh, config)
        search_tool = MemorySearchTool(mesh, config)

        await store_tool.run(content="Memory detail A", actor="client-x")
        await search_tool.run(query="detail A", actor="client-x")

        assert policy.tracer is not None
        records = policy.tracer.records
        kinds = [r.kind for r in records]

        assert EventKind.MEMORY_WRITE in kinds
        assert EventKind.MEMORY_READ in kinds

        write_rec = next(r for r in records if r.kind == EventKind.MEMORY_WRITE)
        assert write_rec.actor == "client-x"
        assert write_rec.payload["name"] == "Memory detail A"

        read_rec = next(r for r in records if r.kind == EventKind.MEMORY_READ)
        assert read_rec.actor == "client-x"
        assert read_rec.payload["query"] == "detail A"
        assert "hit_count" in read_rec.payload
        assert "detail A" not in json.dumps(read_rec.payload.get("hits", ""))

    asyncio.run(run())


def test_export_audit_bundle_produces_verifiable_receipt(tmp_path: Path) -> None:
    async def run() -> None:
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        mesh = KnowledgeMesh(
            MeshConfig(
                roots=[memory_dir],
                state_dir=tmp_path / "state",
                vector_backend="memory",
                graph_backend="memory",
                use_git=False,
            )
        )

        policy = GuardrailPolicy()
        config = MemoryMCPConfig(memory_root=memory_dir, policy=policy)

        store_tool = MemoryStoreTool(mesh, config)
        search_tool = MemorySearchTool(mesh, config)

        await store_tool.run(content="Some audit note", name="Audit Note")
        await search_tool.run(query="audit note")

        bundle_path = str(tmp_path / "audit_bundle.json")
        policy.export_audit_bundle(bundle_path)
        assert Path(bundle_path).exists()

        ver_res = verify(bundle_path, trusted_keys=_manifest_keys_as_trusted(bundle_path))
        assert ver_res.verdict == Verdict.MATCH
        assert ver_res.ok is True

    asyncio.run(run())


def test_build_mesh_tools_rejects_memory_root_outside_mesh_roots(tmp_path: Path) -> None:
    root1 = tmp_path / "root1"
    root1.mkdir()
    root2 = tmp_path / "root2"
    root2.mkdir()

    mesh = KnowledgeMesh(
        MeshConfig(
            roots=[root1],
            state_dir=tmp_path / "state",
            vector_backend="memory",
            graph_backend="memory",
            use_git=False,
        )
    )

    config = MemoryMCPConfig(memory_root=root2)
    with pytest.raises(ValueError, match="is not one of this mesh's configured roots"):
        build_mesh_tools(mesh, config)


def test_backward_compatible_build_mesh_tools_default_args(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    mesh = KnowledgeMesh(
        MeshConfig(
            roots=[root],
            state_dir=tmp_path / "state",
            vector_backend="memory",
            graph_backend="memory",
            use_git=False,
        )
    )

    tools = build_mesh_tools(mesh)
    tool_names = [t.name for t in tools]

    expected = [
        "mesh_query",
        "mesh_reindex",
        "mesh_duplicates",
        "mesh_status",
        "memory_search",
        "memory_recall",
        "memory_store",
    ]
    for name in expected:
        assert name in tool_names


def test_mcp_server_still_works_with_extended_mesh_tools(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    mesh = KnowledgeMesh(
        MeshConfig(
            roots=[root],
            state_dir=tmp_path / "state",
            vector_backend="memory",
            graph_backend="memory",
            use_git=False,
        )
    )

    registered: dict[str, Any] = {}
    mock_server_inst = MagicMock()

    def decorator_factory(key: str):
        def decorator(fn: Any) -> Any:
            registered[key] = fn
            return fn

        return lambda: decorator

    mock_server_inst.list_tools = decorator_factory("list_tools")
    mock_server_inst.call_tool = decorator_factory("call_tool")
    mock_server_inst.list_resources = decorator_factory("list_resources")
    mock_server_inst.read_resource = decorator_factory("read_resource")

    mock_server_mod = MagicMock()
    mock_server_mod.Server = MagicMock(return_value=mock_server_inst)
    mock_types = MagicMock()
    mock_types.TextContent = MagicMock(side_effect=lambda type, text: (type, text))
    mock_types.Tool = MagicMock()
    mock_types.Resource = MagicMock()

    with patch.dict(sys.modules, {"mcp.server": mock_server_mod, "mcp.types": mock_types}):
        MCPServer(mesh)._build_server()

    listed = asyncio.run(registered["list_tools"]())
    assert len(listed) == 7

    # Execute memory_store tool via call_tool handler
    res_store = asyncio.run(
        registered["call_tool"](
            name="memory_store",
            arguments={"content": "Server integration test", "name": "Server Test"},
        )
    )
    assert "stored" in res_store[0][1]

    # Execute memory_search tool via call_tool handler
    res_search = asyncio.run(
        registered["call_tool"](
            name="memory_search",
            arguments={"query": "Server integration"},
        )
    )
    assert "hits" in res_search[0][1]
