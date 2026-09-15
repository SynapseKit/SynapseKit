"""Box file loader using the Box Content API."""

from __future__ import annotations

import asyncio
import os
from typing import Any
from urllib.parse import quote

from ._http_utils import http_client
from ._record_utils import response_json
from .base import Document


class BoxLoader:
    """Load files from a Box folder, optionally descending into subfolders."""

    def __init__(
        self,
        access_token: str | None = None,
        folder_id: str = "0",
        token: str | None = None,
        recursive: bool = False,
        limit: int | None = None,
        endpoint_url: str = "https://api.box.com/2.0",
        client: Any | None = None,
        timeout: float = 30.0,
    ) -> None:
        access_token = access_token or token
        if not access_token:
            raise ValueError("access_token must be provided")
        if not folder_id:
            raise ValueError("folder_id must be provided")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be greater than 0")

        self._access_token = access_token
        self._folder_id = folder_id
        self._recursive = recursive
        self._limit = limit
        self._endpoint_url = endpoint_url.rstrip("/")
        self._client = client
        self._timeout = timeout

    def load(self) -> list[Document]:
        headers = {"Authorization": f"Bearer {self._access_token}"}
        pending = [self._folder_id]
        files: list[dict[str, Any]] = []
        with http_client(self._client, self._timeout, "box") as client:
            while pending and (self._limit is None or len(files) < self._limit):
                folder_id = pending.pop(0)
                offset = 0
                while self._limit is None or len(files) < self._limit:
                    page_size = min(self._limit or 100, 1000)
                    payload = response_json(
                        client.get(
                            f"{self._endpoint_url}/folders/{quote(str(folder_id), safe='')}/items",
                            params={"limit": page_size, "offset": offset},
                            headers=headers,
                        )
                    )
                    entries = payload.get("entries", [])
                    for entry in entries:
                        if entry.get("type") == "folder" and self._recursive:
                            pending.append(str(entry["id"]))
                        elif entry.get("type") == "file":
                            files.append(entry)
                            if self._limit is not None and len(files) >= self._limit:
                                break
                    if self._limit is not None and len(files) >= self._limit:
                        break
                    total_count = payload.get("total_count")
                    if (
                        not entries
                        or (isinstance(total_count, int) and offset + len(entries) >= total_count)
                        or (total_count is None and len(entries) < page_size)
                    ):
                        break
                    offset += len(entries)

            documents: list[Document] = []
            for index, file_data in enumerate(files[: self._limit]):
                content_response = client.get(
                    f"{self._endpoint_url}/files/{quote(str(file_data['id']), safe='')}/content",
                    headers=headers,
                )
                content_response.raise_for_status()
                content = getattr(content_response, "content", b"")
                content_type = content_response.headers.get("Content-Type")
                text, raw_content = self._extract_text(
                    file_data.get("name", ""), content, content_type
                )
                metadata = {
                    # Spread the Box entry first so its fields can never
                    # clobber the loader's own reserved metadata keys
                    # (a Box field literally named "source"/"row"/etc.).
                    **file_data,
                    "source": "box",
                    "row": index,
                    "file_id": file_data.get("id"),
                    "content_type": content_type,
                }
                if raw_content is not None:
                    metadata["raw_content"] = raw_content
                documents.append(Document(text=text, metadata=metadata))
        return documents

    @staticmethod
    def _extract_text(
        name: str, content: Any, content_type: str | None
    ) -> tuple[str, bytes | None]:
        if not isinstance(content, bytes):
            return str(content), None
        try:
            return content.decode("utf-8"), None
        except UnicodeDecodeError:
            ext = os.path.splitext(name)[1].lower()
            descriptor = content_type or ext or "unknown"
            return f"[Binary file: {descriptor}]", content

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)
