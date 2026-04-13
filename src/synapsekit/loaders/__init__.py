from typing import Any

from .azure_blob import AzureBlobLoader
from .base import Document
from .markdown import MarkdownLoader
from .mongodb import MongoDBLoader
from .s3 import S3Loader
from .text import StringLoader, TextLoader

__all__ = [
    "ArXivLoader",
    "AudioLoader",
    "AzureBlobLoader",
    "CSVLoader",
    "ConfigLoader",
    "ConfluenceLoader",
    "DirectoryLoader",
    "DiscordLoader",
    "DocxLoader",
    "Document",
    "DropboxLoader",
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
    "S3Loader",
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
    "AzureBlobLoader": ".azure_blob",
    "PDFLoader": ".pdf",
    "HTMLLoader": ".html",
    "CSVLoader": ".csv",
    "ConfigLoader": ".config",
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
    "S3Loader": ".s3",
    "SQLLoader": ".sql",
    "SupabaseLoader": ".supabase",
    "TeamsLoader": ".teams",
    "WikipediaLoader": ".wikipedia",
    "ConfluenceLoader": ".confluence",
    "MongoDBLoader": ".mongodb",
    "DropboxLoader": ".dropbox",
}


def __getattr__(name: str) -> Any:
    if name in _LOADERS:
        import importlib

        mod = importlib.import_module(_LOADERS[name], __name__)
        cls = getattr(mod, name)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
