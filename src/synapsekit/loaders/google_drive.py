"""GoogleDriveLoader — load files and folders from Google Drive."""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Any

from .base import Document

logger = logging.getLogger(__name__)


class GoogleDriveLoader:
    """Load files and folders from Google Drive into Documents.

    This loader uses the Google Drive API v3 to fetch files by ID or all files
    from a folder. It supports both synchronous and asynchronous loading.

    Prerequisites:
        - Google Cloud project with Drive API enabled
        - Service account credentials (JSON file or dict)
        - Service account must have access to the target files/folders

    Supported file types:
        - Google Docs (exported as text)
        - Google Sheets (exported as CSV)
        - PDFs (downloaded as bytes)
        - Text files (downloaded as text)

    Example::

        loader = GoogleDriveLoader(
            credentials_path="service-account.json",
            file_id="1abc...",  # or folder_id="1def..."
        )
        docs = loader.load()  # synchronous
        # or
        docs = await loader.aload()  # asynchronous
    """

    def __init__(
        self,
        credentials_path: str | None = None,
        credentials_dict: dict[str, Any] | None = None,
        file_id: str | None = None,
        folder_id: str | None = None,
    ) -> None:
        if not credentials_path and not credentials_dict:
            raise ValueError("Either credentials_path or credentials_dict must be provided")
        if file_id and folder_id:
            raise ValueError("Provide either file_id or folder_id, not both")
        if not file_id and not folder_id:
            raise ValueError("Either file_id or folder_id must be provided")

        self.credentials_path = credentials_path
        self.credentials_dict = credentials_dict
        self.file_id = file_id
        self.folder_id = folder_id

    def load(self) -> list[Document]:
        """Synchronously fetch files from Google Drive and return them as Documents."""
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.aload())
        finally:
            loop.close()

    async def aload(self) -> list[Document]:
        """Asynchronously fetch files from Google Drive and return them as Documents."""
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
        except ImportError:
            raise ImportError(
                "Google Drive dependencies required: pip install synapsekit[gdrive]"
            ) from None

        # Build credentials
        if self.credentials_path:
            creds = service_account.Credentials.from_service_account_file(
                self.credentials_path,
                scopes=["https://www.googleapis.com/auth/drive.readonly"],
            )
        else:
            creds = service_account.Credentials.from_service_account_info(
                self.credentials_dict,
                scopes=["https://www.googleapis.com/auth/drive.readonly"],
            )

        loop = asyncio.get_running_loop()
        service = await loop.run_in_executor(None, lambda: build("drive", "v3", credentials=creds))

        if self.file_id:
            return await self._load_file(service, self.file_id)
        else:
            return await self._load_folder(service, self.folder_id)  # type: ignore[arg-type]

    async def _load_file(self, service: Any, file_id: str) -> list[Document]:
        """Load a single file by ID."""
        loop = asyncio.get_running_loop()

        file_metadata = await loop.run_in_executor(
            None,
            lambda: (
                service.files()
                .get(fileId=file_id, fields="id,name,mimeType,modifiedTime")
                .execute()
            ),
        )

        text = await self._download_file(service, file_id, file_metadata["mimeType"])

        return [
            Document(
                text=text,
                metadata={
                    "source": "google_drive",
                    "file_name": file_metadata["name"],
                    "mime_type": file_metadata["mimeType"],
                    "modified": file_metadata["modifiedTime"],
                    "file_id": file_id,
                },
            )
        ]

    async def _load_folder(self, service: Any, folder_id: str) -> list[Document]:
        """Load all files from a folder."""
        loop = asyncio.get_running_loop()

        query = f"'{folder_id}' in parents and trashed=false"
        results = await loop.run_in_executor(
            None,
            lambda: (
                service.files()
                .list(q=query, fields="files(id,name,mimeType,modifiedTime)")
                .execute()
            ),
        )

        files = results.get("files", [])
        documents = []

        for file_data in files:
            if file_data["mimeType"] == "application/vnd.google-apps.folder":
                continue

            try:
                text = await self._download_file(service, file_data["id"], file_data["mimeType"])
                documents.append(
                    Document(
                        text=text,
                        metadata={
                            "source": "google_drive",
                            "file_name": file_data["name"],
                            "mime_type": file_data["mimeType"],
                            "modified": file_data["modifiedTime"],
                            "file_id": file_data["id"],
                        },
                    )
                )
            except Exception as exc:
                logger.warning(
                    "GoogleDriveLoader: skipping file %r (%s) — %s",
                    file_data["name"],
                    file_data["id"],
                    exc,
                )

        return documents

    async def _download_file(self, service: Any, file_id: str, mime_type: str) -> str:
        """Download or export a file based on its MIME type."""
        from googleapiclient.http import MediaIoBaseDownload

        loop = asyncio.get_running_loop()

        if mime_type == "application/vnd.google-apps.document":
            content: bytes = await loop.run_in_executor(
                None,
                lambda: (
                    service.files().export_media(fileId=file_id, mimeType="text/plain").execute()
                ),
            )
            return content.decode("utf-8")

        elif mime_type == "application/vnd.google-apps.spreadsheet":
            content = await loop.run_in_executor(
                None,
                lambda: service.files().export_media(fileId=file_id, mimeType="text/csv").execute(),
            )
            return content.decode("utf-8")

        else:
            request = service.files().get_media(fileId=file_id)
            fh = io.BytesIO()

            def download() -> bytes:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
                return fh.getvalue()

            content = await loop.run_in_executor(None, download)

            try:
                return content.decode("utf-8")
            except UnicodeDecodeError:
                return f"[Binary file: {mime_type}]"
