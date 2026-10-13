import sys
import types

import pytest

from synapsekit.llm.base import BaseLLM, LLMConfig
from synapsekit.loaders.vision_pdf import VisionPDFLoader


class DummyVLM(BaseLLM):
    def __init__(self):
        super().__init__(LLMConfig(model="dummy-vision", api_key="fake", provider="openai"))

    async def generate_with_messages(self, messages, **kw):
        return "```markdown\n# Dummy Markdown\n| Col1 | Col2 |\n|---|---|\n| A | B |\n```"

    async def stream(self, prompt, **kw):
        yield "dummy"


class FakePixmap:
    def tobytes(self, fmt: str) -> bytes:
        return b"fake-png-data"


class FakePage:
    def get_pixmap(self, dpi: int) -> FakePixmap:
        return FakePixmap()


class FakeFitzDocument:
    """Hand-written fake standing in for a fitz.Document (2 pages)."""

    def __init__(self, num_pages: int = 2):
        self._num_pages = num_pages
        self.closed = False

    def __len__(self) -> int:
        return self._num_pages

    def load_page(self, index: int) -> FakePage:
        return FakePage()

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_fitz_module(monkeypatch):
    """Install a hand-written fake `fitz` module so the test doesn't need PyMuPDF installed."""
    fake_doc = FakeFitzDocument(num_pages=2)
    module = types.SimpleNamespace(open=lambda path: fake_doc)
    monkeypatch.setitem(sys.modules, "fitz", module)
    return fake_doc


@pytest.fixture
def pdf_path(tmp_path):
    path = tmp_path / "dummy.pdf"
    path.write_bytes(b"%PDF-1.4 fake content for VisionPDFLoader tests")
    return str(path)


@pytest.mark.asyncio
async def test_vision_pdf_loader_async(fake_fitz_module, pdf_path):
    llm = DummyVLM()
    loader = VisionPDFLoader(pdf_path, llm=llm, max_concurrency=2)

    docs = await loader.aload()

    assert len(docs) == 2
    assert fake_fitz_module.closed
    for i, doc in enumerate(docs):
        assert "Dummy Markdown" in doc.text
        assert doc.metadata["source"] == pdf_path
        assert doc.metadata["page"] == i + 1
        assert doc.metadata["loader"] == "VisionPDFLoader"


def test_vision_pdf_loader_sync(fake_fitz_module, pdf_path):
    llm = DummyVLM()
    loader = VisionPDFLoader(pdf_path, llm=llm)

    docs = loader.load()
    assert len(docs) == 2
    assert "Dummy Markdown" in docs[0].text


def test_vision_pdf_loader_missing_file(fake_fitz_module, tmp_path):
    loader = VisionPDFLoader(str(tmp_path / "missing.pdf"), llm=DummyVLM())
    with pytest.raises(FileNotFoundError):
        loader.load()
