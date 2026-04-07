"""GCSLoader — load files from Google Cloud Storage."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .base import Document

logger = logging.getLogger(__name__)


class GCSLoader:
    """Load files from Google Cloud Storage into Documents.

    This loader uses the Google Cloud Storage Python client to fetch files
    from a bucket. It supports filtering by prefix and handles both
    synchronous and asynchronous loading.

    Prerequisites:
        - Google Cloud project with Storage API enabled
        - Service account credentials (JSON file or dict) or default credentials
        - Service account must have Storage Object Viewer permissions

    Supported file types:
        - Text files (decoded as UTF-8)
        - Binary files (marked as binary in metadata)

    Example::

        loader = GCSLoader(
            bucket_name="my-bucket",
            prefix="documents/",
            credentials_path="service-account.json",
        )
        docs = loader.load()  # synchronous
        # or
        docs = await loader.aload()  # asynchronous
    """

    def __init__(
        self,
        bucket_name: str,
        prefix: str | None = None,
        credentials_path: str | None = None,
        credentials_dict: dict[str, Any] | None = None,
        max_files: int | None = None,
    ) -> None:
        """Initialize GCS loader.

        Args:
            bucket_name: Name of the GCS bucket
            prefix: Optional prefix to filter files (e.g., "folder/subfolder/")
            credentials_path: Path to service account JSON file
            credentials_dict: Service account credentials as dict
            max_files: Maximum number of files to load (None = no limit)
        """
        if not bucket_name:
            raise ValueError("bucket_name must be provided")

        self.bucket_name = bucket_name
        self.prefix = prefix
        self.credentials_path = credentials_path
        self.credentials_dict = credentials_dict
        self.max_files = max_files

    def load(self) -> list[Document]:
        """Synchronously fetch files from GCS and return them as Documents."""
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.aload())
        finally:
            loop.close()

    async def aload(self) -> list[Document]:
        """Asynchronously fetch files from GCS and return them as Documents."""
        try:
            from google.cloud import storage
            from google.oauth2 import service_account
        except ImportError:
            raise ImportError(
                "Google Cloud Storage dependencies required: pip install synapsekit[gcs]"
            ) from None

        loop = asyncio.get_running_loop()

        # Build client with credentials
        if self.credentials_path:
            creds = service_account.Credentials.from_service_account_file(self.credentials_path)
            client = await loop.run_in_executor(None, lambda: storage.Client(credentials=creds))
        elif self.credentials_dict:
            creds = service_account.Credentials.from_service_account_info(self.credentials_dict)
            client = await loop.run_in_executor(None, lambda: storage.Client(credentials=creds))
        else:
            # Use default credentials
            client = await loop.run_in_executor(None, storage.Client)

        bucket = client.bucket(self.bucket_name)

        # List blobs with optional prefix
        blobs = await loop.run_in_executor(
            None, lambda: list(client.list_blobs(bucket, prefix=self.prefix))
        )

        # Apply max_files limit
        if self.max_files is not None:
            blobs = blobs[: self.max_files]

        documents = []

        for blob in blobs:
            # Skip directories (blobs ending with /)
            if blob.name.endswith("/"):
                continue

            try:
                content = await self._download_blob(loop, blob)
                documents.append(
                    Document(
                        text=content,
                        metadata={
                            "source": "gcs",
                            "bucket": self.bucket_name,
                            "file_name": blob.name,
                            "content_type": blob.content_type,
                            "size": blob.size,
                            "updated": blob.updated.isoformat() if blob.updated else None,
                        },
                    )
                )
            except Exception as exc:
                logger.warning("GCSLoader: skipping file %r — %s", blob.name, exc)

        return documents

    async def _download_blob(self, loop: Any, blob: Any) -> str:
        """Download blob content and decode if possible."""
        content_bytes = await loop.run_in_executor(None, blob.download_as_bytes)

        try:
            return content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # For binary files, return a placeholder
            return f"[Binary file: {blob.content_type or 'unknown'}]"
