"""OneDriveLoader — load files from OneDrive or SharePoint via Microsoft Graph."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from typing import Any, cast
from urllib.request import Request, urlopen

from .base import Document

logger = logging.getLogger(__name__)


class OneDriveLoader:
    """Load files from OneDrive or SharePoint into Documents.

    Uses Microsoft Graph drive endpoints and supports folder traversal
    with optional extension filtering.
    """

    _GRAPH_BASE = "https://graph.microsoft.com/v1.0"

    _TEXT_EXTENSIONS = {
        ".txt",
        ".md",
        ".yaml",
        ".yml",
        ".xml",
        ".log",
        ".rst",
        ".py",
    }

    _EXTRACTABLE_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".pptx", ".csv", ".json", ".html", ".htm"}

    def __init__(
        self,
        access_token: str,
        drive_id: str,
        folder_id: str | None = None,
        file_extensions: list[str] | None = None,
        max_files: int | None = None,
        recursive: bool = True,
    ) -> None:
        if not access_token:
            raise ValueError("access_token must be provided")
        if not drive_id:
            raise ValueError("drive_id must be provided")

        self.access_token = access_token
        self.drive_id = drive_id
        self.folder_id = folder_id
        self.file_extensions = self._normalize_extensions(file_extensions)
        self.max_files = max_files
        self.recursive = recursive

    def load(self) -> list[Document]:
        """Synchronously fetch files from OneDrive/SharePoint."""
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.aload())
        finally:
            loop.close()

    async def aload(self) -> list[Document]:
        """Asynchronously fetch files from OneDrive/SharePoint."""
        loop = asyncio.get_running_loop()
        queue: list[str] = [self._children_url(self.folder_id)]
        documents: list[Document] = []

        while queue:
            page_url: str | None = queue.pop(0)

            while page_url:
                page_data = await loop.run_in_executor(None, self._graph_get_json, page_url)
                for item in page_data.get("value", []):
                    if self.max_files is not None and len(documents) >= self.max_files:
                        return documents

                    item_id = item.get("id")
                    if not item_id:
                        continue

                    # Folder item
                    if item.get("folder") is not None:
                        if self.recursive:
                            queue.append(self._children_url(item_id))
                        continue

                    name = item.get("name", "")
                    if self.file_extensions and not self._matches_extension(name):
                        continue

                    mime_type = self._item_mime_type(item)
                    try:
                        content = await loop.run_in_executor(
                            None,
                            self._graph_get_bytes,
                            self._content_url(item_id),
                        )
                        text = self._extract_text(name, content, mime_type)
                        documents.append(
                            Document(
                                text=text,
                                metadata={
                                    "source": "onedrive",
                                    "drive_id": self.drive_id,
                                    "item_id": item_id,
                                    "file_name": name,
                                    "mime_type": mime_type,
                                    "size": item.get("size"),
                                    "web_url": item.get("webUrl"),
                                    "last_modified": item.get("lastModifiedDateTime"),
                                },
                            )
                        )
                    except Exception as exc:
                        logger.warning(
                            "OneDriveLoader: skipping item %r (%s) — %s", name, item_id, exc
                        )

                page_url = page_data.get("@odata.nextLink")

        return documents

    def _children_url(self, folder_id: str | None) -> str:
        if folder_id:
            return f"{self._GRAPH_BASE}/drives/{self.drive_id}/items/{folder_id}/children"
        return f"{self._GRAPH_BASE}/drives/{self.drive_id}/root/children"

    def _content_url(self, item_id: str) -> str:
        return f"{self._GRAPH_BASE}/drives/{self.drive_id}/items/{item_id}/content"

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.access_token}"}

    def _graph_get_json(self, url: str) -> dict[str, Any]:
        req = Request(url=url, headers=self._auth_headers(), method="GET")
        with urlopen(req, timeout=30) as response:
            payload = cast(bytes, response.read())
        return cast(dict[str, Any], json.loads(payload.decode("utf-8")))

    def _graph_get_bytes(self, url: str) -> bytes:
        req = Request(url=url, headers=self._auth_headers(), method="GET")
        with urlopen(req, timeout=30) as response:
            return cast(bytes, response.read())

    @staticmethod
    def _item_mime_type(item: dict[str, Any]) -> str | None:
        file_info = item.get("file")
        if not isinstance(file_info, dict):
            return None
        mime = file_info.get("mimeType")
        return mime if isinstance(mime, str) else None

    def _extract_text(self, filename: str, content: bytes, mime_type: str | None) -> str:
        ext = os.path.splitext(filename)[1].lower()

        if ext in self._TEXT_EXTENSIONS:
            return content.decode("utf-8", errors="replace")

        extracted = self._extract_with_supported_loader(filename, content, ext)
        if extracted is not None:
            return extracted

        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            descriptor = mime_type or ext or "unknown"
            return f"[Binary file: {descriptor}]"

    def _extract_with_supported_loader(self, filename: str, content: bytes, ext: str) -> str | None:
        if ext not in self._EXTRACTABLE_EXTENSIONS:
            return None

        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(content)
                temp_path = tmp.name

            docs = self._run_loader_for_extension(ext, temp_path)
            return "\n\n".join(doc.text for doc in docs if doc.text)
        except Exception as exc:
            logger.warning("OneDriveLoader: extraction fallback for %r — %s", filename, exc)
            return None
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def _run_loader_for_extension(self, ext: str, path: str) -> list[Document]:
        if ext == ".pdf":
            from .pdf import PDFLoader

            return PDFLoader(path).load()
        if ext == ".docx":
            from .docx import DocxLoader

            return DocxLoader(path).load()
        if ext == ".xlsx":
            from .excel import ExcelLoader

            return ExcelLoader(path).load()
        if ext == ".pptx":
            from .pptx import PowerPointLoader

            return PowerPointLoader(path).load()
        if ext == ".csv":
            from .csv import CSVLoader

            return CSVLoader(path).load()
        if ext == ".json":
            from .json_loader import JSONLoader

            return JSONLoader(path).load()
        if ext in {".html", ".htm"}:
            from .html import HTMLLoader

            return HTMLLoader(path).load()

        return []

    @staticmethod
    def _normalize_extensions(file_extensions: list[str] | None) -> set[str] | None:
        if file_extensions is None:
            return None

        normalized: set[str] = set()
        for ext in file_extensions:
            ext_clean = ext.strip().lower()
            if not ext_clean:
                continue
            if not ext_clean.startswith("."):
                ext_clean = f".{ext_clean}"
            normalized.add(ext_clean)
        return normalized

    def _matches_extension(self, filename: str) -> bool:
        ext = os.path.splitext(filename)[1].lower()
        return bool(self.file_extensions and ext in self.file_extensions)
