from __future__ import annotations

from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from synapsekit.loaders.base import Document
from synapsekit.loaders.rtf import RTFLoader

RTF_CONTENT = r"{\rtf1\ansi{\fonttbl\f0\fswiss Helvetica;}\f0\pard Hello World\par}"


def _fake_striprtf(return_value: str) -> dict[str, ModuleType]:
    """Inject a minimal fake striprtf module tree into sys.modules."""
    pkg = ModuleType("striprtf")
    sub = ModuleType("striprtf.striprtf")
    sub.rtf_to_text = MagicMock(return_value=return_value)  # type: ignore[attr-defined]
    pkg.striprtf = sub  # type: ignore[attr-defined]
    return {"striprtf": pkg, "striprtf.striprtf": sub}


class TestRTFLoader:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            RTFLoader("/nonexistent/file.rtf").load()

    def test_missing_dependency_raises(self, tmp_path):
        p = tmp_path / "doc.rtf"
        p.write_text(RTF_CONTENT)
        with patch.dict("sys.modules", {"striprtf": None, "striprtf.striprtf": None}):
            with pytest.raises(ImportError, match="striprtf required"):
                RTFLoader(str(p)).load()

    def test_returns_list_of_documents(self, tmp_path):
        p = tmp_path / "doc.rtf"
        p.write_text(RTF_CONTENT)
        with patch.dict("sys.modules", _fake_striprtf("Hello World")):
            docs = RTFLoader(str(p)).load()
        assert isinstance(docs, list)
        assert len(docs) == 1
        assert isinstance(docs[0], Document)

    def test_metadata_contains_source(self, tmp_path):
        p = tmp_path / "doc.rtf"
        p.write_text(RTF_CONTENT)
        with patch.dict("sys.modules", _fake_striprtf("Hello World")):
            docs = RTFLoader(str(p)).load()
        assert docs[0].metadata["source"] == str(p)

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.rtf"
        p.write_text("")
        with patch.dict("sys.modules", _fake_striprtf("")):
            docs = RTFLoader(str(p)).load()
        assert len(docs) == 1
        assert docs[0].text == ""

    def test_text_extraction(self, tmp_path):
        p = tmp_path / "doc.rtf"
        p.write_text(RTF_CONTENT)
        with patch.dict("sys.modules", _fake_striprtf("Hello World")):
            docs = RTFLoader(str(p)).load()
        assert "Hello World" in docs[0].text

    def test_text_is_stripped(self, tmp_path):
        p = tmp_path / "doc.rtf"
        p.write_text(RTF_CONTENT)
        with patch.dict("sys.modules", _fake_striprtf("  Hello World  \n")):
            docs = RTFLoader(str(p)).load()
        assert docs[0].text == "Hello World"

    def test_malformed_rtf_does_not_raise(self, tmp_path):
        p = tmp_path / "bad.rtf"
        p.write_text("not valid rtf content {{{ \\\\")
        with patch.dict("sys.modules", _fake_striprtf("some text")):
            docs = RTFLoader(str(p)).load()
        assert len(docs) == 1

    async def test_aload(self, tmp_path):
        p = tmp_path / "doc.rtf"
        p.write_text(RTF_CONTENT)
        with patch.dict("sys.modules", _fake_striprtf("Hello World")):
            docs = await RTFLoader(str(p)).aload()
        assert len(docs) == 1
        assert docs[0].metadata["source"] == str(p)
