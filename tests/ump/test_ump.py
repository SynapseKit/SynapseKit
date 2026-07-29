from __future__ import annotations

import inspect
from pathlib import Path

from synapsekit.ump import (
    AiderAdapter,
    ClaudeAdapter,
    ContinueAdapter,
    CursorAdapter,
    UMPDocument,
    UMPFrontmatter,
    UMPProvenance,
    UMPReader,
    UMPValidator,
    UMPWriter,
    auto_detect_and_convert,
)


def test_ump_provenance_defaults() -> None:
    prov = UMPProvenance()
    assert prov.authors == []
    assert prov.evidence == []
    assert prov.signed_by == ""


def test_ump_frontmatter_defaults() -> None:
    fm = UMPFrontmatter()
    assert fm.ump_version == "1.0"
    assert fm.name == ""
    assert fm.type == "general"
    assert fm.scope == "project"
    assert fm.visibility == "local"


def test_ump_document_roundtrip_dict() -> None:
    doc = UMPDocument(
        frontmatter=UMPFrontmatter(name="test-doc", type="user"),
        body="## Body content\nTest body",
        source_path="/tmp/test.md",
    )
    as_dict = doc.to_dict()
    reconstructed = UMPDocument.from_dict(as_dict)
    assert reconstructed.frontmatter.name == "test-doc"
    assert reconstructed.frontmatter.type == "user"
    assert reconstructed.body == "## Body content\nTest body"


def test_reader_parse_valid_ump() -> None:
    raw = """---
ump_version: "1.0"
name: test-memory
type: feedback
scope: project
visibility: local
provenance:
  authors: ["human", "agent:claude"]
links: ["[[testing-standards]]"]
---

Memory body text with wikilink [[user-preferences]].
"""
    doc = UMPReader.parse(raw)
    assert doc.frontmatter.name == "test-memory"
    assert doc.frontmatter.type == "feedback"
    assert doc.frontmatter.provenance.authors == ["human", "agent:claude"]
    assert "[[testing-standards]]" in doc.frontmatter.links
    assert "[[user-preferences]]" in doc.frontmatter.links


def test_reader_parse_no_frontmatter() -> None:
    raw = "# Plain Markdown\nNo frontmatter here."
    doc = UMPReader.parse(raw)
    assert doc.frontmatter.ump_version == "1.0"
    assert doc.body == "# Plain Markdown\nNo frontmatter here."


def test_writer_serialize_and_parse_roundtrip() -> None:
    original = UMPDocument(
        frontmatter=UMPFrontmatter(
            name="roundtrip-test",
            type="project",
            provenance=UMPProvenance(authors=["human"]),
            links=["[[ref-1]]"],
        ),
        body="Roundtrip body content.",
    )
    serialized = UMPWriter.serialize(original)
    parsed = UMPReader.parse(serialized)

    assert parsed.frontmatter.name == original.frontmatter.name
    assert parsed.frontmatter.type == original.frontmatter.type
    assert parsed.body.strip() == original.body.strip()


async def test_writer_and_reader_file_io(tmp_path: Path) -> None:
    file_path = tmp_path / "memory.ump.md"
    doc = UMPDocument(
        frontmatter=UMPFrontmatter(name="file-io-test"),
        body="Stored on disk.",
    )
    await UMPWriter.write_file(doc, file_path)

    assert file_path.exists()
    loaded = await UMPReader.read_file(file_path)
    assert loaded.frontmatter.name == "file-io-test"
    assert loaded.body.strip() == "Stored on disk."


async def test_write_file_creates_parent_dirs(tmp_path: Path) -> None:
    file_path = tmp_path / "nested" / "deep" / "memory.ump.md"
    doc = UMPDocument(
        frontmatter=UMPFrontmatter(name="nested-io"),
        body="Nested write.",
    )
    await UMPWriter.write_file(doc, file_path)

    assert file_path.exists()
    loaded = await UMPReader.read_file(file_path)
    assert loaded.frontmatter.name == "nested-io"


def test_validator_valid_document() -> None:
    doc = UMPDocument(
        frontmatter=UMPFrontmatter(name="valid-doc", type="user"),
        body="Valid body content.",
    )
    res = UMPValidator.validate(doc)
    assert res.is_valid
    assert len(res.errors) == 0


def test_validator_invalid_enums() -> None:
    doc = UMPDocument(
        frontmatter=UMPFrontmatter(
            name="invalid-doc",
            type="invalid_type",  # type: ignore
            scope="invalid_scope",  # type: ignore
            visibility="invalid_vis",  # type: ignore
        ),
        body="Body",
    )
    res = UMPValidator.validate(doc)
    assert not res.is_valid
    assert len(res.errors) == 3


async def test_claude_adapter(tmp_path: Path) -> None:
    claude_file = tmp_path / "CLAUDE.md"
    claude_file.write_text("# Project Guidelines\n- Use pytest", encoding="utf-8")

    assert ClaudeAdapter.detect(tmp_path)
    doc = await ClaudeAdapter.to_ump(claude_file)
    assert doc.frontmatter.name == "claude-memory"
    assert doc.frontmatter.provenance.authors == ["agent:claude-code"]
    assert "Project Guidelines" in doc.body


async def test_cursor_adapter(tmp_path: Path) -> None:
    cursor_file = tmp_path / ".cursorrules"
    cursor_file.write_text("Always use strict typing", encoding="utf-8")

    assert CursorAdapter.detect(tmp_path)
    doc = await CursorAdapter.to_ump(cursor_file)
    assert doc.frontmatter.name == "cursor-rules"
    assert "Always use strict typing" in doc.body


async def test_auto_detect_and_convert(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text("# Claude memory", encoding="utf-8")
    (tmp_path / ".cursorrules").write_text("Cursor rules", encoding="utf-8")

    docs = await auto_detect_and_convert(tmp_path)
    assert len(docs) == 2
    names = {d.frontmatter.name for d in docs}
    assert "claude-memory" in names
    assert "cursor-rules" in names


def test_public_file_io_methods_are_coroutines() -> None:
    # Tripwire: these public file-IO methods must stay async so they never
    # block the event loop. Pure string methods (parse/serialize) stay sync.
    assert inspect.iscoroutinefunction(UMPReader.read_file)
    assert inspect.iscoroutinefunction(UMPWriter.write_file)
    assert inspect.iscoroutinefunction(UMPValidator.validate_file)
    assert inspect.iscoroutinefunction(ClaudeAdapter.to_ump)
    assert inspect.iscoroutinefunction(CursorAdapter.to_ump)
    assert inspect.iscoroutinefunction(AiderAdapter.to_ump)
    assert inspect.iscoroutinefunction(ContinueAdapter.to_ump)
    assert inspect.iscoroutinefunction(auto_detect_and_convert)


def test_pure_string_methods_stay_sync() -> None:
    # These operate on in-memory strings only and must remain synchronous.
    assert not inspect.iscoroutinefunction(UMPReader.parse)
    assert not inspect.iscoroutinefunction(UMPWriter.serialize)
    assert not inspect.iscoroutinefunction(UMPValidator.validate)


async def test_reader_malformed_frontmatter_falls_back(tmp_path: Path) -> None:
    # Malformed YAML frontmatter must not crash; parser falls back to defaults.
    file_path = tmp_path / "bad.ump.md"
    file_path.write_text(
        "---\nname: [unclosed\n  bad: : :\n---\n\nBody survives.\n",
        encoding="utf-8",
    )
    doc = await UMPReader.read_file(file_path)
    assert doc.frontmatter.ump_version == "1.0"
    assert "Body survives." in doc.body


async def test_aider_adapter_malformed_yaml_fallback(tmp_path: Path) -> None:
    aider_file = tmp_path / ".aider.conf.yml"
    aider_file.write_text("model: [unterminated\n  : : bad", encoding="utf-8")

    doc = await AiderAdapter.to_ump(aider_file)
    assert doc.frontmatter.name == "aider-config"
    # Falls back to raw text since YAML parsing failed.
    assert "unterminated" in doc.body


async def test_continue_adapter_malformed_json_fallback(tmp_path: Path) -> None:
    cfg_dir = tmp_path / ".continue"
    cfg_dir.mkdir()
    cfg_file = cfg_dir / "config.json"
    cfg_file.write_text("{not valid json,,,}", encoding="utf-8")

    doc = await ContinueAdapter.to_ump(cfg_file)
    assert doc.frontmatter.name == "continue-config"
    # Falls back to raw text since JSON parsing failed.
    assert "not valid json" in doc.body


async def test_adapter_missing_file_yields_empty_body(tmp_path: Path) -> None:
    doc = await ClaudeAdapter.to_ump(tmp_path / "CLAUDE.md")
    assert doc.frontmatter.name == "claude-memory"
    assert doc.body == ""


async def test_validate_file_valid_document(tmp_path: Path) -> None:
    file_path = tmp_path / "valid.ump.md"
    doc = UMPDocument(
        frontmatter=UMPFrontmatter(name="ok", type="user"),
        body="Content.",
    )
    await UMPWriter.write_file(doc, file_path)

    res = await UMPValidator.validate_file(file_path)
    assert res.is_valid
    assert len(res.errors) == 0


async def test_validate_file_invalid_document(tmp_path: Path) -> None:
    file_path = tmp_path / "invalid.ump.md"
    file_path.write_text(
        "---\nname: bad\ntype: not_a_type\nscope: not_a_scope\n"
        "visibility: not_a_vis\n---\n\nBody.\n",
        encoding="utf-8",
    )
    res = await UMPValidator.validate_file(file_path)
    assert not res.is_valid
    assert len(res.errors) == 3


async def test_validate_file_missing_file_returns_error(tmp_path: Path) -> None:
    res = await UMPValidator.validate_file(tmp_path / "does_not_exist.ump.md")
    assert not res.is_valid
    assert any(e.field == "file" for e in res.errors)
