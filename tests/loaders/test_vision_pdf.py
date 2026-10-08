import asyncio
import os
import pytest
from unittest.mock import MagicMock, patch

from synapsekit.llm.base import BaseLLM, LLMConfig
from synapsekit.loaders.vision_pdf import VisionPDFLoader

class DummyVLM(BaseLLM):
    def __init__(self):
        super().__init__(LLMConfig(model="dummy-vision", api_key="fake", provider="openai"))

    async def generate_with_messages(self, messages, **kw):
        return "```markdown\n# Dummy Markdown\n| Col1 | Col2 |\n|---|---|\n| A | B |\n```"

    async def stream(self, prompt, **kw):
        yield "dummy"

@pytest.fixture
def mock_fitz_doc():
    """Mock PyMuPDF document."""
    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 2  # 2 pages
    
    mock_page = MagicMock()
    mock_pix = MagicMock()
    mock_pix.tobytes.return_value = b"fake-png-data"
    mock_page.get_pixmap.return_value = mock_pix
    
    mock_doc.load_page.return_value = mock_page
    return mock_doc

@pytest.mark.asyncio
async def test_vision_pdf_loader_async(mock_fitz_doc, tmp_path):
    # Mock os.path.exists so it doesn't fail on our fake path
    with patch("os.path.exists", return_value=True), \
         patch("mimetypes.guess_type", return_value=("application/pdf", None)), \
         patch.dict("sys.modules", {"fitz": MagicMock(open=MagicMock(return_value=mock_fitz_doc))}):
        
        llm = DummyVLM()
        loader = VisionPDFLoader("dummy.pdf", llm=llm, max_concurrency=2)
        
        docs = await loader.aload()
        
        assert len(docs) == 2
        for i, doc in enumerate(docs):
            assert "Dummy Markdown" in doc.text
            assert doc.metadata["source"] == "dummy.pdf"
            assert doc.metadata["page"] == i + 1
            assert doc.metadata["loader"] == "VisionPDFLoader"

def test_vision_pdf_loader_sync(mock_fitz_doc):
    with patch("os.path.exists", return_value=True), \
         patch("mimetypes.guess_type", return_value=("application/pdf", None)), \
         patch.dict("sys.modules", {"fitz": MagicMock(open=MagicMock(return_value=mock_fitz_doc))}):
        
        llm = DummyVLM()
        loader = VisionPDFLoader("dummy.pdf", llm=llm)
        
        docs = loader.load()
        assert len(docs) == 2
        assert "Dummy Markdown" in docs[0].text
