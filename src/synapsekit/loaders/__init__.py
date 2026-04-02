from typing import Any

from .base import Document
from .markdown import MarkdownLoader
from .text import StringLoader, TextLoader

__all__ = [
    "AudioLoader",
    "CSVLoader",
    "DirectoryLoader",
    "DiscordLoader",
    "DocxLoader",
    "Document",
    "GoogleDriveLoader",
    "HTMLLoader",
    "JSONLoader",
    "MarkdownLoader",
    "NotionLoader",
    "PDFLoader",
    "StringLoader",
    "TextLoader",
    "VideoLoader",
    "WebLoader",
    "XMLLoader",
    "YAMLLoader",
]

_LOADERS = {
    "PDFLoader": ".pdf",
    "HTMLLoader": ".html",
    "CSVLoader": ".csv",
    "JSONLoader": ".json_loader",
    "DirectoryLoader": ".directory",
    "WebLoader": ".web",
    "DocxLoader": ".docx",
    "AudioLoader": ".audio",
    "VideoLoader": ".video",
    "XMLLoader": ".xml_loader",
    "YAMLLoader": ".yaml_loader",
    "DiscordLoader": ".discord",
    "GoogleDriveLoader": ".google_drive",
    "NotionLoader": ".notion",
}


def __getattr__(name: str) -> Any:
    if name in _LOADERS:
        import importlib

        mod = importlib.import_module(_LOADERS[name], __name__)
        cls = getattr(mod, name)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
