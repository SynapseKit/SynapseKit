from __future__ import annotations

import inspect

import pytest

from synapsekit.loaders.base import Document
from synapsekit.loaders.obsidian import ObsidianLoader


class TestObsidianLoader:
    def test_missing_path_raises(self):
        with pytest.raises(FileNotFoundError):
            ObsidianLoader("/nonexistent/vault").load()

    def test_non_directory_path_raises(self, tmp_path):
        file_path = tmp_path / "note.md"
        file_path.write_text("# note")
        with pytest.raises(NotADirectoryError):
            ObsidianLoader(str(file_path)).load()

    def test_load_multiple_markdown_files(self, tmp_path):
        (tmp_path / "a.md").write_text("First #tag")
        (tmp_path / "b.md").write_text("Second [[Another Page]]")
        (tmp_path / "ignore.txt").write_text("not markdown")

        docs = ObsidianLoader(str(tmp_path)).load()

        assert len(docs) == 2
        assert all(isinstance(doc, Document) for doc in docs)
        sources = {doc.metadata["source"] for doc in docs}
        assert str(tmp_path / "a.md") in sources
        assert str(tmp_path / "b.md") in sources

    def test_frontmatter_parsing_and_stripping(self, tmp_path):
        note = tmp_path / "frontmatter.md"
        note.write_text(
            "---\ntitle: My Note\ntags: [ai, ml]\n---\nBody text with #bodytag and [[Page|Alias]]."
        )

        docs = ObsidianLoader(str(tmp_path)).load()

        assert len(docs) == 1
        doc = docs[0]
        assert "title: My Note" not in doc.text
        assert "Body text with #bodytag and [[Page|Alias]]." in doc.text
        assert doc.metadata["frontmatter"] == {"title": "My Note", "tags": ["ai", "ml"]}
        assert doc.metadata["tags"] == ["bodytag"]
        assert doc.metadata["links"] == ["Page"]

    def test_frontmatter_with_leading_whitespace(self, tmp_path):
        note = tmp_path / "leading_frontmatter.md"
        note.write_text("\n\n---\ntitle: Leading\n---\nBody text")

        docs = ObsidianLoader(str(tmp_path)).load()
        doc = docs[0]

        assert doc.metadata["frontmatter"] == {"title": "Leading"}
        assert doc.text == "Body text"

    def test_extracts_tags_and_wikilinks(self, tmp_path):
        note = tmp_path / "links.md"
        note.write_text(
            "See [[Page Name]] and [[Page Name|Shown]]. Topics: #alpha #beta #alpha and #tag123."
        )

        docs = ObsidianLoader(str(tmp_path)).load()
        metadata = docs[0].metadata

        assert metadata["links"] == ["Page Name"]
        assert metadata["tags"] == ["alpha", "beta", "tag123"]

    def test_recursive_loading(self, tmp_path):
        (tmp_path / "root.md").write_text("Root file")
        nested = tmp_path / "nested"
        nested.mkdir()
        (nested / "child.md").write_text("Nested file")

        recursive_docs = ObsidianLoader(str(tmp_path), recursive=True).load()
        non_recursive_docs = ObsidianLoader(str(tmp_path), recursive=False).load()

        assert len(recursive_docs) == 2
        assert len(non_recursive_docs) == 1
        assert non_recursive_docs[0].metadata["source"] == str(tmp_path / "root.md")

    def test_empty_file(self, tmp_path):
        (tmp_path / "empty.md").write_text("")
        docs = ObsidianLoader(str(tmp_path)).load()

        assert docs == []

    def test_malformed_frontmatter_is_ignored(self, tmp_path):
        note = tmp_path / "broken.md"
        note.write_text("---\ntitle: Missing end delimiter\nBody #tag")

        docs = ObsidianLoader(str(tmp_path)).load()
        doc = docs[0]

        assert doc.metadata["frontmatter"] == {}
        assert doc.text == "---\ntitle: Missing end delimiter\nBody #tag"
        assert doc.metadata["tags"] == ["tag"]

    async def test_aload(self, tmp_path):
        (tmp_path / "async.md").write_text("Async note #ok")
        docs = await ObsidianLoader(str(tmp_path)).aload()

        assert len(docs) == 1
        assert docs[0].metadata["tags"] == ["ok"]

    def test_aload_is_coroutine_function(self):
        assert inspect.iscoroutinefunction(ObsidianLoader("vault").aload)
