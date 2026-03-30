"""Tests for GoogleDriveLoader."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synapsekit.loaders.base import Document
from synapsekit.loaders.google_drive import GoogleDriveLoader


class TestGoogleDriveLoaderValidation:
    def test_credentials_required(self):
        with pytest.raises(ValueError, match="credentials"):
            GoogleDriveLoader(file_id="123")

    def test_file_or_folder_required(self):
        with pytest.raises(ValueError, match="file_id or folder_id"):
            GoogleDriveLoader(credentials_path="creds.json")

    def test_file_and_folder_mutually_exclusive(self):
        with pytest.raises(ValueError, match="not both"):
            GoogleDriveLoader(
                credentials_path="creds.json", file_id="123", folder_id="456"
            )

    def test_valid_construction_with_file_id(self):
        loader = GoogleDriveLoader(credentials_path="creds.json", file_id="123")
        assert loader.file_id == "123"
        assert loader.folder_id is None

    def test_valid_construction_with_folder_id(self):
        loader = GoogleDriveLoader(credentials_path="creds.json", folder_id="456")
        assert loader.folder_id == "456"
        assert loader.file_id is None

    def test_credentials_dict_allowed(self):
        loader = GoogleDriveLoader(
            credentials_dict={"type": "service_account"}, file_id="123"
        )
        assert loader.credentials_dict == {"type": "service_account"}
        assert loader.credentials_path is None


class TestGoogleDriveLoaderImport:
    def test_missing_dependencies_raises_import_error(self):
        import sys

        loader = GoogleDriveLoader(credentials_path="creds.json", file_id="123")
        with patch.dict(sys.modules, {"google": None, "googleapiclient": None}):
            with pytest.raises(ImportError, match="pip install synapsekit\\[gdrive\\]"):
                loader.load()


class TestGoogleDriveLoaderFileLoading:
    def _make_mock_service(self, file_metadata: dict, file_content: bytes) -> MagicMock:
        """Create a mock Google Drive service."""
        service = MagicMock()

        # Mock files().get()
        get_mock = MagicMock()
        get_mock.execute.return_value = file_metadata
        service.files.return_value.get.return_value = get_mock

        # Mock files().get_media() for downloads
        media_mock = MagicMock()
        service.files.return_value.get_media.return_value = media_mock

        # Mock files().export_media() for Google Docs/Sheets
        export_mock = MagicMock()
        export_mock.execute.return_value = file_content
        service.files.return_value.export_media.return_value = export_mock

        return service

    @pytest.mark.asyncio
    async def test_load_google_doc(self):
        file_metadata = {
            "id": "doc123",
            "name": "Test Document.gdoc",
            "mimeType": "application/vnd.google-apps.document",
            "modifiedTime": "2026-03-30T10:00:00Z",
        }
        file_content = b"This is a Google Doc"

        service = self._make_mock_service(file_metadata, file_content)

        # Mock the google modules
        mock_creds = MagicMock()
        mock_service_account = MagicMock()
        mock_service_account.Credentials.from_service_account_file.return_value = mock_creds

        mock_google = MagicMock()
        mock_google.oauth2.service_account = mock_service_account

        mock_googleapiclient = MagicMock()
        mock_googleapiclient.discovery.build.return_value = service
        # Mock MediaIoBaseDownload (not used for Google Docs, but imported)
        mock_googleapiclient.http.MediaIoBaseDownload = MagicMock()

        loader = GoogleDriveLoader(credentials_path="creds.json", file_id="doc123")

        with patch.dict("sys.modules", {
            "google": mock_google,
            "google.oauth2": mock_google.oauth2,
            "google.oauth2.service_account": mock_service_account,
            "googleapiclient": mock_googleapiclient,
            "googleapiclient.discovery": mock_googleapiclient.discovery,
            "googleapiclient.http": mock_googleapiclient.http,
        }):
            docs = await loader.aload()

        assert len(docs) == 1
        assert docs[0].text == "This is a Google Doc"
        assert docs[0].metadata["source"] == "google_drive"
        assert docs[0].metadata["file_name"] == "Test Document.gdoc"
        assert docs[0].metadata["mime_type"] == "application/vnd.google-apps.document"
        assert docs[0].metadata["file_id"] == "doc123"

    @pytest.mark.asyncio
    async def test_load_google_sheet(self):
        file_metadata = {
            "id": "sheet123",
            "name": "Test Sheet.gsheet",
            "mimeType": "application/vnd.google-apps.spreadsheet",
            "modifiedTime": "2026-03-30T11:00:00Z",
        }
        file_content = b"Name,Age\nAlice,30\nBob,25\n"

        service = self._make_mock_service(file_metadata, file_content)

        # Mock the google modules
        mock_creds = MagicMock()
        mock_service_account = MagicMock()
        mock_service_account.Credentials.from_service_account_file.return_value = mock_creds

        mock_google = MagicMock()
        mock_google.oauth2.service_account = mock_service_account

        mock_googleapiclient = MagicMock()
        mock_googleapiclient.discovery.build.return_value = service
        # Mock MediaIoBaseDownload (not used for Google Sheets, but imported)
        mock_googleapiclient.http.MediaIoBaseDownload = MagicMock()

        loader = GoogleDriveLoader(credentials_path="creds.json", file_id="sheet123")

        with patch.dict("sys.modules", {
            "google": mock_google,
            "google.oauth2": mock_google.oauth2,
            "google.oauth2.service_account": mock_service_account,
            "googleapiclient": mock_googleapiclient,
            "googleapiclient.discovery": mock_googleapiclient.discovery,
            "googleapiclient.http": mock_googleapiclient.http,
        }):
            docs = await loader.aload()

        assert len(docs) == 1
        assert "Alice,30" in docs[0].text
        assert docs[0].metadata["mime_type"] == "application/vnd.google-apps.spreadsheet"

    @pytest.mark.asyncio
    async def test_load_text_file(self):
        file_metadata = {
            "id": "txt123",
            "name": "notes.txt",
            "mimeType": "text/plain",
            "modifiedTime": "2026-03-30T12:00:00Z",
        }
        file_content = b"Plain text content"

        service = MagicMock()
        get_mock = MagicMock()
        get_mock.execute.return_value = file_metadata
        service.files.return_value.get.return_value = get_mock

        # Mock MediaIoBaseDownload for regular files
        def mock_download(fh, request):
            downloader = MagicMock()
            downloader.next_chunk.side_effect = [
                (None, False),
                (None, True),
            ]
            fh.write(file_content)
            return downloader

        # Mock the google modules
        mock_creds = MagicMock()
        mock_service_account = MagicMock()
        mock_service_account.Credentials.from_service_account_file.return_value = mock_creds

        mock_google = MagicMock()
        mock_google.oauth2.service_account = mock_service_account

        mock_googleapiclient = MagicMock()
        mock_googleapiclient.discovery.build.return_value = service
        mock_googleapiclient.http.MediaIoBaseDownload.side_effect = mock_download

        loader = GoogleDriveLoader(credentials_path="creds.json", file_id="txt123")

        with patch.dict("sys.modules", {"google": mock_google, "google.oauth2": mock_google.oauth2, "google.oauth2.service_account": mock_service_account, "googleapiclient": mock_googleapiclient, "googleapiclient.discovery": mock_googleapiclient.discovery, "googleapiclient.http": mock_googleapiclient.http}):
            docs = await loader.aload()

        assert len(docs) == 1
        assert "Plain text content" in docs[0].text
        assert docs[0].metadata["file_name"] == "notes.txt"


class TestGoogleDriveLoaderFolderLoading:
    @pytest.mark.asyncio
    async def test_load_folder(self):
        folder_list_result = {
            "files": [
                {
                    "id": "file1",
                    "name": "doc1.gdoc",
                    "mimeType": "application/vnd.google-apps.document",
                    "modifiedTime": "2026-03-30T10:00:00Z",
                },
                {
                    "id": "file2",
                    "name": "doc2.gdoc",
                    "mimeType": "application/vnd.google-apps.document",
                    "modifiedTime": "2026-03-30T11:00:00Z",
                },
            ]
        }

        service = MagicMock()

        # Mock files().list()
        list_mock = MagicMock()
        list_mock.execute.return_value = folder_list_result
        service.files.return_value.list.return_value = list_mock

        # Mock files().export_media() for exports
        export_mock = MagicMock()
        export_mock.execute.side_effect = [
            b"Content of doc1",
            b"Content of doc2",
        ]
        service.files.return_value.export_media.return_value = export_mock

        # Mock the google modules
        mock_creds = MagicMock()
        mock_service_account = MagicMock()
        mock_service_account.Credentials.from_service_account_file.return_value = mock_creds

        mock_google = MagicMock()
        mock_google.oauth2.service_account = mock_service_account

        mock_googleapiclient = MagicMock()
        mock_googleapiclient.discovery.build.return_value = service
        mock_googleapiclient.http.MediaIoBaseDownload = MagicMock()

        loader = GoogleDriveLoader(credentials_path="creds.json", folder_id="folder123")

        with patch.dict("sys.modules", {
            "google": mock_google,
            "google.oauth2": mock_google.oauth2,
            "google.oauth2.service_account": mock_service_account,
            "googleapiclient": mock_googleapiclient,
            "googleapiclient.discovery": mock_googleapiclient.discovery,
            "googleapiclient.http": mock_googleapiclient.http,
        }):
            docs = await loader.aload()

        assert len(docs) == 2
        assert docs[0].metadata["file_name"] == "doc1.gdoc"
        assert docs[1].metadata["file_name"] == "doc2.gdoc"

    @pytest.mark.asyncio
    async def test_load_folder_skips_subfolders(self):
        folder_list_result = {
            "files": [
                {
                    "id": "file1",
                    "name": "doc1.gdoc",
                    "mimeType": "application/vnd.google-apps.document",
                    "modifiedTime": "2026-03-30T10:00:00Z",
                },
                {
                    "id": "subfolder",
                    "name": "Subfolder",
                    "mimeType": "application/vnd.google-apps.folder",
                    "modifiedTime": "2026-03-30T09:00:00Z",
                },
            ]
        }

        service = MagicMock()

        list_mock = MagicMock()
        list_mock.execute.return_value = folder_list_result
        service.files.return_value.list.return_value = list_mock

        export_mock = MagicMock()
        export_mock.execute.return_value = b"Content of doc1"
        service.files.return_value.export_media.return_value = export_mock

        # Mock the google modules
        mock_creds = MagicMock()
        mock_service_account = MagicMock()
        mock_service_account.Credentials.from_service_account_file.return_value = mock_creds

        mock_google = MagicMock()
        mock_google.oauth2.service_account = mock_service_account

        mock_googleapiclient = MagicMock()
        mock_googleapiclient.discovery.build.return_value = service
        mock_googleapiclient.http.MediaIoBaseDownload = MagicMock()

        loader = GoogleDriveLoader(credentials_path="creds.json", folder_id="folder123")

        with patch.dict("sys.modules", {
            "google": mock_google,
            "google.oauth2": mock_google.oauth2,
            "google.oauth2.service_account": mock_service_account,
            "googleapiclient": mock_googleapiclient,
            "googleapiclient.discovery": mock_googleapiclient.discovery,
            "googleapiclient.http": mock_googleapiclient.http,
        }):
            docs = await loader.aload()

        # Should only get the document, not the folder
        assert len(docs) == 1
        assert docs[0].metadata["file_name"] == "doc1.gdoc"

    @pytest.mark.asyncio
    async def test_load_empty_folder(self):
        service = MagicMock()

        list_mock = MagicMock()
        list_mock.execute.return_value = {"files": []}
        service.files.return_value.list.return_value = list_mock

        # Mock the google modules
        mock_creds = MagicMock()
        mock_service_account = MagicMock()
        mock_service_account.Credentials.from_service_account_file.return_value = mock_creds

        mock_google = MagicMock()
        mock_google.oauth2.service_account = mock_service_account

        mock_googleapiclient = MagicMock()
        mock_googleapiclient.discovery.build.return_value = service
        mock_googleapiclient.http.MediaIoBaseDownload = MagicMock()

        loader = GoogleDriveLoader(credentials_path="creds.json", folder_id="folder123")

        with patch.dict("sys.modules", {
            "google": mock_google,
            "google.oauth2": mock_google.oauth2,
            "google.oauth2.service_account": mock_service_account,
            "googleapiclient": mock_googleapiclient,
            "googleapiclient.discovery": mock_googleapiclient.discovery,
            "googleapiclient.http": mock_googleapiclient.http,
        }):
            docs = await loader.aload()

        assert docs == []


class TestGoogleDriveLoaderSync:
    def test_load_sync_wraps_async(self):
        expected = [Document(text="sync content", metadata={"source": "google_drive"})]

        loader = GoogleDriveLoader(credentials_path="creds.json", file_id="123")

        # Mock the google modules
        mock_google = MagicMock()
        mock_googleapiclient = MagicMock()

        with patch.dict("sys.modules", {"google": mock_google, "google.oauth2": mock_google.oauth2, "googleapiclient": mock_googleapiclient, "googleapiclient.discovery": mock_googleapiclient.discovery}):
            with patch.object(
                loader, "aload", new=AsyncMock(return_value=expected)
            ):
                result = loader.load()

        assert result == expected
