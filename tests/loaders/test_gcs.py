"""Tests for GCSLoader."""

from __future__ import annotations

import pytest

from synapsekit.loaders import GCSLoader


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

    def test_init_with_credentials_dict(self):
        """Test initialization with credentials dictionary."""
        creds = {"type": "service_account", "project_id": "test"}
        loader = GCSLoader(bucket_name="my-bucket", credentials_dict=creds)
        assert loader.bucket_name == "my-bucket"
        assert loader.credentials_dict == creds

    @pytest.mark.asyncio
    async def test_aload_missing_dependencies(self):
        """Test error when GCS dependencies are missing."""
        import sys
        from unittest.mock import patch

        def mock_import(name, *args, **kwargs):
            if "google" in name:
                raise ImportError("No module named 'google'")
            return __import__(name, *args, **kwargs)

        with patch.dict(sys.modules, {"google.cloud.storage": None}):
            with patch("builtins.__import__", side_effect=mock_import):
                loader = GCSLoader(bucket_name="test-bucket")

                with pytest.raises(
                    ImportError,
                    match="Google Cloud Storage dependencies required",
                ):
                    await loader.aload()
