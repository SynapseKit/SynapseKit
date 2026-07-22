from __future__ import annotations

from pathlib import Path

from synapsekit.ump import (
    ClaudeAdapter,
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


def test_writer_and_reader_file_io(tmp_path: Path) -> None:
    file_path = tmp_path / "memory.ump.md"
    doc = UMPDocument(
        frontmatter=UMPFrontmatter(name="file-io-test"),
        body="Stored on disk.",
    )
    UMPWriter.write_file(doc, file_path)

    assert file_path.exists()
    loaded = UMPReader.read_file(file_path)
    assert loaded.frontmatter.name == "file-io-test"
    assert loaded.body.strip() == "Stored on disk."


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


def test_claude_adapter(tmp_path: Path) -> None:
    claude_file = tmp_path / "CLAUDE.md"
    claude_file.write_text("# Project Guidelines\n- Use pytest", encoding="utf-8")

    assert ClaudeAdapter.detect(tmp_path)
    doc = ClaudeAdapter.to_ump(claude_file)
    assert doc.frontmatter.name == "claude-memory"
    assert doc.frontmatter.provenance.authors == ["agent:claude-code"]
    assert "Project Guidelines" in doc.body


def test_cursor_adapter(tmp_path: Path) -> None:
    cursor_file = tmp_path / ".cursorrules"
    cursor_file.write_text("Always use strict typing", encoding="utf-8")

    assert CursorAdapter.detect(tmp_path)
    doc = CursorAdapter.to_ump(cursor_file)
    assert doc.frontmatter.name == "cursor-rules"
    assert "Always use strict typing" in doc.body


def test_auto_detect_and_convert(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text("# Claude memory", encoding="utf-8")
    (tmp_path / ".cursorrules").write_text("Cursor rules", encoding="utf-8")

    docs = auto_detect_and_convert(tmp_path)
    assert len(docs) == 2
    names = {d.frontmatter.name for d in docs}
    assert "claude-memory" in names
    assert "cursor-rules" in names
