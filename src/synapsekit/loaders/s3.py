"""S3Loader — load files from Amazon S3."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from typing import Any, cast

from .base import Document

logger = logging.getLogger(__name__)


class S3Loader:
    """Load files from an Amazon S3 bucket into Documents."""

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
        bucket_name: str,
        prefix: str | None = None,
        file_extensions: list[str] | None = None,
        region_name: str = "us-east-1",
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        endpoint_url: str | None = None,
        max_files: int | None = None,
    ) -> None:
        if not bucket_name:
            raise ValueError("bucket_name must be provided")

        self.bucket_name = bucket_name
        self.prefix = prefix
        self.region_name = region_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        # Custom endpoint for S3-compatible stores (MinIO, Cloudflare R2,
        # LocalStack, …). Left None for real AWS S3.
        self.endpoint_url = endpoint_url
        self.max_files = max_files
        self.file_extensions = self._normalize_extensions(file_extensions)

    def load(self) -> list[Document]:
        """Synchronously fetch files from S3."""
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.aload())
        finally:
            loop.close()

    async def aload(self) -> list[Document]:
        """Asynchronously fetch files from S3."""
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 required: pip install synapsekit[s3]") from None

        loop = asyncio.get_running_loop()
        client = await loop.run_in_executor(None, self._build_client, boto3)
        objects = await loop.run_in_executor(None, self._list_objects, client)

        docs: list[Document] = []

        for obj in objects:
            key = cast(str, obj["Key"])
            if key.endswith("/"):
                continue
            if self.file_extensions and not self._matches_extension(key):
                continue

            try:
                content_bytes, content_type = await loop.run_in_executor(
                    None,
                    self._download_object,
                    client,
                    key,
                )
                text = self._extract_text(key, content_bytes, content_type)
                docs.append(
                    Document(
                        text=text,
                        metadata={
                            "source": "s3",
                            "bucket": self.bucket_name,
                            "key": key,
                            "last_modified": (
                                obj["LastModified"].isoformat() if obj.get("LastModified") else None
                            ),
                            "size": obj.get("Size"),
                            "content_type": content_type,
                        },
                    )
                )
            except Exception as exc:
                logger.warning("S3Loader: skipping object %r — %s", key, exc)

            if self.max_files is not None and len(docs) >= self.max_files:
                break

        return docs

    def _build_client(self, boto3_module: Any) -> Any:
        kwargs: dict[str, Any] = {"region_name": self.region_name}
        if self.aws_access_key_id:
            kwargs["aws_access_key_id"] = self.aws_access_key_id
        if self.aws_secret_access_key:
            kwargs["aws_secret_access_key"] = self.aws_secret_access_key
        if self.aws_session_token:
            kwargs["aws_session_token"] = self.aws_session_token
        if self.endpoint_url:
            kwargs["endpoint_url"] = self.endpoint_url
            # S3-compatible endpoints addressed by host:port need path-style
            # addressing (virtual-host style can't resolve a bare host/IP).
            from botocore.config import Config

            kwargs["config"] = Config(s3={"addressing_style": "path"})
        return boto3_module.client("s3", **kwargs)

    def _list_objects(self, client: Any) -> list[dict[str, Any]]:
        paginator = client.get_paginator("list_objects_v2")
        paginate_kwargs: dict[str, Any] = {"Bucket": self.bucket_name}
        if self.prefix:
            paginate_kwargs["Prefix"] = self.prefix

        objects: list[dict[str, Any]] = []
        for page in paginator.paginate(**paginate_kwargs):
            objects.extend(cast(list[dict[str, Any]], page.get("Contents", [])))
        return objects

    def _download_object(self, client: Any, key: str) -> tuple[bytes, str | None]:
        response = cast(dict[str, Any], client.get_object(Bucket=self.bucket_name, Key=key))
        body = response.get("Body")
        if body is None:
            raise ValueError(f"S3 object has no body: {key}")
        content = cast(bytes, body.read())
        return content, cast(str | None, response.get("ContentType"))

    def _extract_text(self, key: str, content: bytes, content_type: str | None) -> str:
        ext = os.path.splitext(key)[1].lower()

        if ext in self._TEXT_EXTENSIONS:
            return content.decode("utf-8", errors="replace")

        extracted = self._extract_with_supported_loader(key, content, ext)
        if extracted is not None:
            return extracted

        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            descriptor = content_type or ext or "unknown"
            return f"[Binary file: {descriptor}]"

    def _extract_with_supported_loader(self, key: str, content: bytes, ext: str) -> str | None:
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
            logger.warning("S3Loader: extraction fallback for %r — %s", key, exc)
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

    def _matches_extension(self, key: str) -> bool:
        ext = os.path.splitext(key)[1].lower()
        return bool(self.file_extensions and ext in self.file_extensions)
