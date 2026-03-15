from .base import Document
from .markdown import MarkdownLoader
from .text import StringLoader, TextLoader

__all__ = [
    "CSVLoader",
    "DirectoryLoader",
    "DocxLoader",
    "Document",
    "HTMLLoader",
    "JSONLoader",
    "MarkdownLoader",
    "PDFLoader",
    "StringLoader",
    "TextLoader",
    "WebLoader",
]

_LOADERS = {
    "PDFLoader": ".pdf",
    "HTMLLoader": ".html",
    "CSVLoader": ".csv",
    "JSONLoader": ".json_loader",
    "DirectoryLoader": ".directory",
    "WebLoader": ".web",
    "DocxLoader": ".docx",
}


def __getattr__(name: str):
    if name in _LOADERS:
        import importlib

        mod = importlib.import_module(_LOADERS[name], __name__)
        cls = getattr(mod, name)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
