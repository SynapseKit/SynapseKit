from typing import Any

from .azure_blob import AzureBlobLoader
from .base import Document
from .bigquery import BigQueryLoader
from .hubspot import HubSpotLoader
from .markdown import MarkdownLoader
from .mongodb import MongoDBLoader
from .onedrive import OneDriveLoader
from .s3 import S3Loader
from .salesforce import SalesforceLoader
from .text import StringLoader, TextLoader

__all__ = [
    "AirtableLoader",
    "ArXivLoader",
    "AudioLoader",
    "AzureBlobLoader",
    "BigQueryLoader",
    "CSVLoader",
    "ConfigLoader",
    "ConfluenceLoader",
    "DirectoryLoader",
    "DiscordLoader",
    "DocxLoader",
    "Document",
    "DynamoDBLoader",
    "DropboxLoader",
    "EPUBLoader",
    "ElasticsearchLoader",
    "EmailLoader",
    "GCSLoader",
    "GitHubLoader",
    "GitLoader",
    "GoogleDriveLoader",
    "GoogleSheetsLoader",
    "HTMLLoader",
    "HubSpotLoader",
    "LaTeXLoader",
    "JSONLoader",
    "JiraLoader",
    "MarkdownLoader",
    "MongoDBLoader",
    "NotionLoader",
    "ObsidianLoader",
    "OneDriveLoader",
    "PDFLoader",
    "ParquetLoader",
    "RSSLoader",
    "RTFLoader",
    "RedisLoader",
    "S3Loader",
    "SalesforceLoader",
    "SitemapLoader",
    "SnowflakeLoader",
    "SQLLoader",
    "SlackLoader",
    "StringLoader",
    "SupabaseLoader",
    "TeamsLoader",
    "TextLoader",
    "TSVLoader",
    "VideoLoader",
    "WebLoader",
    "WikipediaLoader",
    "XMLLoader",
    "YAMLLoader",
    "YouTubeLoader",
]

_LOADERS = {
    "AirtableLoader": ".airtable",
    "ArXivLoader": ".arxiv",
    "AzureBlobLoader": ".azure_blob",
    "BigQueryLoader": ".bigquery",
    "PDFLoader": ".pdf",
    "HTMLLoader": ".html",
    "LaTeXLoader": ".latex",
    "CSVLoader": ".csv",
    "TSVLoader": ".tsv",
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
    "HubSpotLoader": ".hubspot",
    "JiraLoader": ".jira",
    "SlackLoader": ".slack",
    "NotionLoader": ".notion",
    "ObsidianLoader": ".obsidian",
    "OneDriveLoader": ".onedrive",
    "RSSLoader": ".rss",
    "RTFLoader": ".rtf",
    "S3Loader": ".s3",
    "SalesforceLoader": ".salesforce",
    "SQLLoader": ".sql",
    "SnowflakeLoader": ".snowflake",
    "SupabaseLoader": ".supabase",
    "TeamsLoader": ".teams",
    "WikipediaLoader": ".wikipedia",
    "ConfluenceLoader": ".confluence",
    "MongoDBLoader": ".mongodb",
    "DynamoDBLoader": ".dynamodb",
    "DropboxLoader": ".dropbox",
    "EPUBLoader": ".epub",
    "ParquetLoader": ".parquet",
    "RedisLoader": ".redis_loader",
    "ElasticsearchLoader": ".elasticsearch",
    "SitemapLoader": ".sitemap",
    "YouTubeLoader": ".youtube",
}


def __getattr__(name: str) -> Any:
    if name in _LOADERS:
        import importlib

        mod = importlib.import_module(_LOADERS[name], __name__)
        cls = getattr(mod, name)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
