from __future__ import annotations

import pytest

from synapsekit.loaders.base import Document
from synapsekit.loaders.tsv import TSVLoader

TSV_CONTENT = "name\tage\tcity\nAlice\t30\tNew York\nBob\t25\tLondon\n"


class TestTSVLoader:
    def test_file_not_found(self):
        loader = TSVLoader("/nonexistent/file.tsv")
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_returns_list_of_documents(self, tmp_path):
        p = tmp_path / "data.tsv"
        p.write_text(TSV_CONTENT)
        docs = TSVLoader(str(p)).load()
        assert isinstance(docs, list)
        assert all(isinstance(d, Document) for d in docs)

    def test_multiple_rows(self, tmp_path):
        p = tmp_path / "data.tsv"
        p.write_text(TSV_CONTENT)
        docs = TSVLoader(str(p)).load()
        assert len(docs) == 2

    def test_empty_file_returns_empty_list(self, tmp_path):
        p = tmp_path / "empty.tsv"
        p.write_text("")
        docs = TSVLoader(str(p)).load()
        assert docs == []

    def test_header_only_file_returns_empty_list(self, tmp_path):
        p = tmp_path / "header.tsv"
        p.write_text("name\tage\tcity\n")
        docs = TSVLoader(str(p)).load()
        assert docs == []

    def test_metadata_source(self, tmp_path):
        p = tmp_path / "data.tsv"
        p.write_text(TSV_CONTENT)
        docs = TSVLoader(str(p)).load()
        assert all(d.metadata["source"] == str(p) for d in docs)

    def test_metadata_row_index(self, tmp_path):
        p = tmp_path / "data.tsv"
        p.write_text(TSV_CONTENT)
        docs = TSVLoader(str(p)).load()
        assert docs[0].metadata["row"] == 0
        assert docs[1].metadata["row"] == 1

    def test_default_text_joins_values(self, tmp_path):
        p = tmp_path / "data.tsv"
        p.write_text(TSV_CONTENT)
        docs = TSVLoader(str(p)).load()
        assert "Alice" in docs[0].text
        assert "30" in docs[0].text
        assert "New York" in docs[0].text

    def test_text_column(self, tmp_path):
        p = tmp_path / "data.tsv"
        p.write_text(TSV_CONTENT)
        docs = TSVLoader(str(p), text_column="name").load()
        assert docs[0].text == "Alice"
        assert docs[1].text == "Bob"

    def test_text_column_excluded_from_meta(self, tmp_path):
        p = tmp_path / "data.tsv"
        p.write_text(TSV_CONTENT)
        docs = TSVLoader(str(p), text_column="name").load()
        assert "name" not in docs[0].metadata

    def test_text_column_missing_key_returns_empty(self, tmp_path):
        p = tmp_path / "data.tsv"
        p.write_text(TSV_CONTENT)
        docs = TSVLoader(str(p), text_column="nonexistent").load()
        assert docs[0].text == ""

    def test_skips_empty_rows(self, tmp_path):
        p = tmp_path / "empty_rows.tsv"
        p.write_text("name\tage\nAlice\t30\n\t\nBob\t25\n")
        docs = TSVLoader(str(p)).load()
        assert len(docs) == 2

    def test_no_none_strings_in_text(self, tmp_path):
        p = tmp_path / "none.tsv"
        p.write_text("name\tvalue\nAlice\t\n")
        docs = TSVLoader(str(p)).load()
        assert "None" not in docs[0].text

    def test_single_column(self, tmp_path):
        p = tmp_path / "single.tsv"
        p.write_text("title\nFoo\nBar\n")
        docs = TSVLoader(str(p)).load()
        assert len(docs) == 2
        assert "Foo" in docs[0].text

    def test_tab_delimiter_not_comma(self, tmp_path):
        p = tmp_path / "tabs.tsv"
        p.write_text("a\tb\n1\t2\n")
        docs = TSVLoader(str(p)).load()
        assert len(docs) == 1
        assert "1" in docs[0].text
        assert "2" in docs[0].text

    async def test_aload(self, tmp_path):
        p = tmp_path / "data.tsv"
        p.write_text(TSV_CONTENT)
        docs = await TSVLoader(str(p)).aload()
        assert len(docs) == 2
        assert docs[0].metadata["source"] == str(p)

    def test_aload_is_coroutine_function(self):
        import inspect

        assert inspect.iscoroutinefunction(TSVLoader("/tmp/x.tsv").aload)
