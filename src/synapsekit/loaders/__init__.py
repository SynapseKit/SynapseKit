from typing import Any

from .base import Document
from .markdown import MarkdownLoader
from .mongodb import MongoDBLoader
from .text import StringLoader, TextLoader

__all__ = [
    "ArXivLoader",
    "AudioLoader",
    "CSVLoader",
    "ConfluenceLoader",
    "DirectoryLoader",
    "DiscordLoader",
    "DocxLoader",
    "Document",
    "EmailLoader",
    "GCSLoader",
    "GitHubLoader",
    "GitLoader",
    "GoogleDriveLoader",
    "GoogleSheetsLoader",
    "HTMLLoader",
    "JSONLoader",
    "JiraLoader",
    "MarkdownLoader",
    "MongoDBLoader",
    "NotionLoader",
    "PDFLoader",
    "RSSLoader",
    "SQLLoader",
    "SlackLoader",
    "StringLoader",
    "SupabaseLoader",
    "TeamsLoader",
    "TextLoader",
    "VideoLoader",
    "WebLoader",
    "WikipediaLoader",
    "XMLLoader",
    "YAMLLoader",
]

_LOADERS = {
    "ArXivLoader": ".arxiv",
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
    "EmailLoader": ".email",
    "GCSLoader": ".gcs",
    "GitHubLoader": ".github",
    "GitLoader": ".git",
    "GoogleDriveLoader": ".google_drive",
    "GoogleSheetsLoader": ".google_sheets",
    "JiraLoader": ".jira",
    "SlackLoader": ".slack",
    "NotionLoader": ".notion",
    "RSSLoader": ".rss",
    "SQLLoader": ".sql",
    "SupabaseLoader": ".supabase",
    "TeamsLoader": ".teams",
    "WikipediaLoader": ".wikipedia",
    "ConfluenceLoader": ".confluence",
    "MongoDBLoader": ".mongodb",
}


def __getattr__(name: str) -> Any:
    if name in _LOADERS:
        import importlib

        mod = importlib.import_module(_LOADERS[name], __name__)
        cls = getattr(mod, name)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
