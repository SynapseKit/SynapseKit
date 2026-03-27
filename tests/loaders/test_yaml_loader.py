from __future__ import annotations

import pytest

from synapsekit.loaders.base import Document
from synapsekit.loaders.yaml_loader import YAMLLoader


class TestYAMLLoader:
    def test_file_not_found(self):
        loader = YAMLLoader("/nonexistent/file.yaml")
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_load_list_of_dicts(self, tmp_path):
        p = tmp_path / "docs.yaml"
        p.write_text("- text: Hello world\n- text: Second doc\n")
        loader = YAMLLoader(str(p))
        docs = loader.load()

        assert len(docs) == 2
        assert isinstance(docs[0], Document)
        assert docs[0].text == "Hello world"
        assert docs[1].text == "Second doc"

    def test_load_single_dict(self, tmp_path):
        p = tmp_path / "single.yaml"
        p.write_text("text: Only one doc\ntitle: My Title\n")
        loader = YAMLLoader(str(p))
        docs = loader.load()

        assert len(docs) == 1
        assert docs[0].text == "Only one doc"

    def test_metadata_source_and_index(self, tmp_path):
        p = tmp_path / "meta.yaml"
        p.write_text("- text: Doc one\n- text: Doc two\n")
        loader = YAMLLoader(str(p))
        docs = loader.load()

        assert docs[0].metadata["source"] == str(p)
        assert docs[0].metadata["index"] == 0
        assert docs[1].metadata["index"] == 1

    def test_custom_text_key(self, tmp_path):
        p = tmp_path / "custom.yaml"
        p.write_text("- content: Custom key content\n")
        loader = YAMLLoader(str(p), text_key="content")
        docs = loader.load()

        assert docs[0].text == "Custom key content"

    def test_metadata_keys_extracted(self, tmp_path):
        p = tmp_path / "with_meta.yaml"
        p.write_text("- text: Hello\n  author: Alice\n  category: news\n")
        loader = YAMLLoader(str(p), metadata_keys=["author", "category"])
        docs = loader.load()

        assert docs[0].metadata["author"] == "Alice"
        assert docs[0].metadata["category"] == "news"

    def test_missing_text_key_returns_empty_string(self, tmp_path):
        p = tmp_path / "notext.yaml"
        p.write_text("- title: No text key here\n")
        loader = YAMLLoader(str(p))
        docs = loader.load()

        assert docs[0].text == ""

    def test_top_level_export(self):
        from synapsekit import YAMLLoader as ImportedLoader

        assert ImportedLoader is YAMLLoader
