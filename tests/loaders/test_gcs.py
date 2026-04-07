"""Tests for GCSLoader."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synapsekit.loaders import Document, GCSLoader


@pytest.fixture
def mock_blob():
    """Create a mock GCS blob."""
    blob = MagicMock()
    blob.name = "test-file.txt"
    blob.content_type = "text/plain"
    blob.size = 1234
    blob.updated = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
    blob.download_as_bytes = MagicMock(return_value=b"Test content from GCS")
    return blob


@pytest.fixture
def mock_binary_blob():
    """Create a mock binary blob."""
    blob = MagicMock()
    blob.name = "test-image.png"
    blob.content_type = "image/png"
    blob.size = 5678
    blob.updated = datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc)
    blob.download_as_bytes = MagicMock(return_value=b"\x89PNG\r\n\x1a\n")
    return blob


@pytest.fixture
def mock_storage_client():
    """Mock Google Cloud Storage client."""
    with patch("synapsekit.loaders.gcs.storage") as mock_storage:
        yield mock_storage


@pytest.fixture
def mock_service_account():
    """Mock service account credentials."""
    with patch("synapsekit.loaders.gcs.service_account") as mock_sa:
        yield mock_sa


class TestGCSLoader:
    """Test suite for GCSLoader."""

    def test_init_requires_bucket_name(self):
        """Test that bucket_name is required."""
        with pytest.raises(ValueError, match="bucket_name must be provided"):
            GCSLoader(bucket_name="")

    def test_init_with_bucket_name(self):
        """Test initialization with bucket name only."""
        loader = GCSLoader(bucket_name="my-bucket")
        assert loader.bucket_name == "my-bucket"
        assert loader.prefix is None
        assert loader.credentials_path is None
        assert loader.credentials_dict is None
        assert loader.max_files is None

    def test_init_with_all_params(self):
        """Test initialization with all parameters."""
        loader = GCSLoader(
            bucket_name="my-bucket",
            prefix="documents/",
            credentials_path="creds.json",
            max_files=10,
        )
        assert loader.bucket_name == "my-bucket"
        assert loader.prefix == "documents/"
        assert loader.credentials_path == "creds.json"
        assert loader.max_files == 10

    @pytest.mark.asyncio
    async def test_aload_single_text_file(
        self, mock_storage_client, mock_service_account, mock_blob
    ):
        """Test loading a single text file."""
        mock_client = MagicMock()
        mock_storage_client.Client.return_value = mock_client
        mock_client.list_blobs.return_value = [mock_blob]

        loader = GCSLoader(bucket_name="test-bucket")
        docs = await loader.aload()

        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert docs[0].text == "Test content from GCS"
        assert docs[0].metadata["source"] == "gcs"
        assert docs[0].metadata["bucket"] == "test-bucket"
        assert docs[0].metadata["file_name"] == "test-file.txt"
        assert docs[0].metadata["content_type"] == "text/plain"
        assert docs[0].metadata["size"] == 1234

    @pytest.mark.asyncio
    async def test_aload_binary_file(
        self, mock_storage_client, mock_service_account, mock_binary_blob
    ):
        """Test loading a binary file."""
        mock_client = MagicMock()
        mock_storage_client.Client.return_value = mock_client
        mock_client.list_blobs.return_value = [mock_binary_blob]

        loader = GCSLoader(bucket_name="test-bucket")
        docs = await loader.aload()

        assert len(docs) == 1
        assert docs[0].text == "[Binary file: image/png]"
        assert docs[0].metadata["file_name"] == "test-image.png"

    @pytest.mark.asyncio
    async def test_aload_with_prefix(self, mock_storage_client, mock_blob):
        """Test loading with prefix filter."""
        mock_client = MagicMock()
        mock_storage_client.Client.return_value = mock_client
        mock_client.list_blobs.return_value = [mock_blob]

        loader = GCSLoader(bucket_name="test-bucket", prefix="folder/")
        await loader.aload()

        mock_client.list_blobs.assert_called_once()
        call_args = mock_client.list_blobs.call_args
        assert call_args[1]["prefix"] == "folder/"

    @pytest.mark.asyncio
    async def test_aload_with_max_files(
        self, mock_storage_client, mock_blob
    ):
        """Test loading with max_files limit."""
        mock_client = MagicMock()
        mock_storage_client.Client.return_value = mock_client

        # Create multiple blobs
        blob1 = MagicMock()
        blob1.name = "file1.txt"
        blob1.content_type = "text/plain"
        blob1.size = 100
        blob1.updated = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        blob1.download_as_bytes = MagicMock(return_value=b"Content 1")

        blob2 = MagicMock()
        blob2.name = "file2.txt"
        blob2.content_type = "text/plain"
        blob2.size = 100
        blob2.updated = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        blob2.download_as_bytes = MagicMock(return_value=b"Content 2")

        blob3 = MagicMock()
        blob3.name = "file3.txt"
        blob3.content_type = "text/plain"
        blob3.size = 100
        blob3.updated = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        blob3.download_as_bytes = MagicMock(return_value=b"Content 3")

        mock_client.list_blobs.return_value = [blob1, blob2, blob3]

        loader = GCSLoader(bucket_name="test-bucket", max_files=2)
        docs = await loader.aload()

        assert len(docs) == 2

    @pytest.mark.asyncio
    async def test_aload_skips_directories(self, mock_storage_client):
        """Test that directory blobs are skipped."""
        mock_client = MagicMock()
        mock_storage_client.Client.return_value = mock_client

        dir_blob = MagicMock()
        dir_blob.name = "folder/"

        file_blob = MagicMock()
        file_blob.name = "file.txt"
        file_blob.content_type = "text/plain"
        file_blob.size = 100
        file_blob.updated = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        file_blob.download_as_bytes = MagicMock(return_value=b"Content")

        mock_client.list_blobs.return_value = [dir_blob, file_blob]

        loader = GCSLoader(bucket_name="test-bucket")
        docs = await loader.aload()

        assert len(docs) == 1
        assert docs[0].metadata["file_name"] == "file.txt"

    @pytest.mark.asyncio
    async def test_aload_with_credentials_path(
        self, mock_storage_client, mock_service_account, mock_blob
    ):
        """Test loading with credentials file path."""
        mock_creds = MagicMock()
        mock_service_account.Credentials.from_service_account_file.return_value = (
            mock_creds
        )

        mock_client = MagicMock()
        mock_storage_client.Client.return_value = mock_client
        mock_client.list_blobs.return_value = [mock_blob]

        loader = GCSLoader(
            bucket_name="test-bucket", credentials_path="service-account.json"
        )
        await loader.aload()

        mock_service_account.Credentials.from_service_account_file.assert_called_once_with(
            "service-account.json"
        )
        mock_storage_client.Client.assert_called_with(credentials=mock_creds)

    @pytest.mark.asyncio
    async def test_aload_with_credentials_dict(
        self, mock_storage_client, mock_service_account, mock_blob
    ):
        """Test loading with credentials dictionary."""
        mock_creds = MagicMock()
        mock_service_account.Credentials.from_service_account_info.return_value = (
            mock_creds
        )

        mock_client = MagicMock()
        mock_storage_client.Client.return_value = mock_client
        mock_client.list_blobs.return_value = [mock_blob]

        creds_dict = {"type": "service_account", "project_id": "test"}
        loader = GCSLoader(bucket_name="test-bucket", credentials_dict=creds_dict)
        await loader.aload()

        mock_service_account.Credentials.from_service_account_info.assert_called_once_with(
            creds_dict
        )

    def test_load_synchronous(self, mock_storage_client, mock_blob):
        """Test synchronous load method."""
        mock_client = MagicMock()
        mock_storage_client.Client.return_value = mock_client
        mock_client.list_blobs.return_value = [mock_blob]

        loader = GCSLoader(bucket_name="test-bucket")
        docs = loader.load()

        assert len(docs) == 1
        assert docs[0].text == "Test content from GCS"

    @pytest.mark.asyncio
    async def test_aload_missing_dependencies(self):
        """Test error when GCS dependencies are missing."""
        with patch("synapsekit.loaders.gcs.storage", None):
            loader = GCSLoader(bucket_name="test-bucket")

            with pytest.raises(
                ImportError,
                match="Google Cloud Storage dependencies required: pip install synapsekit\\[gcs\\]",
            ):
                await loader.aload()
