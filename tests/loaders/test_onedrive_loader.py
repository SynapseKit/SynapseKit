"""Tests for OneDriveLoader."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from synapsekit.loaders import Document, OneDriveLoader


class TestOneDriveLoader:
    def test_init_requires_access_token(self):
        with pytest.raises(ValueError, match="access_token must be provided"):
            OneDriveLoader(access_token="", drive_id="drive123")

    def test_init_requires_drive_id(self):
        with pytest.raises(ValueError, match="drive_id must be provided"):
            OneDriveLoader(access_token="token", drive_id="")

    def test_init_with_defaults(self):
        loader = OneDriveLoader(access_token="token", drive_id="drive123")
        assert loader.drive_id == "drive123"
        assert loader.folder_id is None
        assert loader.file_extensions is None
        assert loader.max_files is None
        assert loader.recursive is True

    def test_init_normalizes_extensions(self):
        loader = OneDriveLoader(
            access_token="token",
            drive_id="drive123",
            file_extensions=["txt", ".PDF", "  .json  "],
        )
        assert loader.file_extensions == {".txt", ".pdf", ".json"}

    def test_load_with_text_and_binary_files(self):
        loader = OneDriveLoader(access_token="token", drive_id="drive123")

        root_children = {
            "value": [
                {
                    "id": "item-1",
                    "name": "notes.txt",
                    "size": 11,
                    "webUrl": "https://example/notes",
                    "lastModifiedDateTime": "2026-04-01T10:00:00Z",
                    "file": {"mimeType": "text/plain"},
                },
                {
                    "id": "item-2",
                    "name": "image.png",
                    "size": 2048,
                    "webUrl": "https://example/image",
                    "lastModifiedDateTime": "2026-04-01T11:00:00Z",
                    "file": {"mimeType": "image/png"},
                },
            ]
        }

        def mock_get_json(url: str):
            assert url.endswith("/drives/drive123/root/children")
            return root_children

        def mock_get_bytes(url: str):
            if url.endswith("/items/item-1/content"):
                return b"hello world"
            return b"\x89PNG\x00\x01"

        with (
            patch.object(loader, "_graph_get_json", side_effect=mock_get_json),
            patch.object(loader, "_graph_get_bytes", side_effect=mock_get_bytes),
        ):
            docs = loader.load()

        assert len(docs) == 2
        assert all(isinstance(d, Document) for d in docs)

        assert docs[0].text == "hello world"
        assert docs[0].metadata["source"] == "onedrive"
        assert docs[0].metadata["drive_id"] == "drive123"
        assert docs[0].metadata["item_id"] == "item-1"
        assert docs[0].metadata["file_name"] == "notes.txt"

        assert docs[1].text == "[Binary file: image/png]"
        assert docs[1].metadata["item_id"] == "item-2"

    def test_load_applies_extension_filters(self):
        loader = OneDriveLoader(
            access_token="token",
            drive_id="drive123",
            file_extensions=[".txt"],
        )

        root_children = {
            "value": [
                {"id": "a", "name": "a.txt", "file": {"mimeType": "text/plain"}},
                {"id": "b", "name": "b.pdf", "file": {"mimeType": "application/pdf"}},
            ]
        }

        with (
            patch.object(loader, "_graph_get_json", return_value=root_children),
            patch.object(loader, "_graph_get_bytes", return_value=b"ok"),
        ):
            docs = loader.load()

        assert len(docs) == 1
        assert docs[0].metadata["file_name"] == "a.txt"

    def test_load_extracts_supported_file_via_loader(self):
        loader = OneDriveLoader(access_token="token", drive_id="drive123")

        root_children = {
            "value": [
                {
                    "id": "pdf-id",
                    "name": "report.pdf",
                    "file": {"mimeType": "application/pdf"},
                }
            ]
        }

        with (
            patch.object(loader, "_graph_get_json", return_value=root_children),
            patch.object(loader, "_graph_get_bytes", return_value=b"%PDF-1.7 fake"),
            patch.object(loader, "_run_loader_for_extension") as run_loader,
        ):
            run_loader.return_value = [
                Document(text="Extracted Page 1", metadata={}),
                Document(text="Extracted Page 2", metadata={}),
            ]
            docs = loader.load()

        assert len(docs) == 1
        assert docs[0].text == "Extracted Page 1\n\nExtracted Page 2"
        run_loader.assert_called_once()

    def test_load_handles_recursive_folder_traversal(self):
        loader = OneDriveLoader(access_token="token", drive_id="drive123", recursive=True)

        root_url = "https://graph.microsoft.com/v1.0/drives/drive123/root/children"
        child_url = "https://graph.microsoft.com/v1.0/drives/drive123/items/folder-1/children"

        def mock_get_json(url: str):
            if url == root_url:
                return {
                    "value": [
                        {"id": "folder-1", "name": "docs", "folder": {"childCount": 1}},
                        {"id": "file-1", "name": "root.txt", "file": {"mimeType": "text/plain"}},
                    ]
                }
            if url == child_url:
                return {
                    "value": [
                        {
                            "id": "file-2",
                            "name": "child.txt",
                            "file": {"mimeType": "text/plain"},
                        }
                    ]
                }
            raise AssertionError(f"unexpected url: {url}")

        def mock_get_bytes(url: str):
            if url.endswith("/items/file-1/content"):
                return b"root"
            return b"child"

        with (
            patch.object(loader, "_graph_get_json", side_effect=mock_get_json),
            patch.object(loader, "_graph_get_bytes", side_effect=mock_get_bytes),
        ):
            docs = loader.load()

        assert len(docs) == 2
        names = {doc.metadata["file_name"] for doc in docs}
        assert names == {"root.txt", "child.txt"}

    def test_load_non_recursive_skips_subfolder_items(self):
        loader = OneDriveLoader(access_token="token", drive_id="drive123", recursive=False)

        root_children = {
            "value": [
                {"id": "folder-1", "name": "docs", "folder": {"childCount": 1}},
                {"id": "file-1", "name": "root.txt", "file": {"mimeType": "text/plain"}},
            ]
        }

        with (
            patch.object(loader, "_graph_get_json", return_value=root_children),
            patch.object(loader, "_graph_get_bytes", return_value=b"root"),
        ):
            docs = loader.load()

        assert len(docs) == 1
        assert docs[0].metadata["file_name"] == "root.txt"

    def test_load_respects_max_files(self):
        loader = OneDriveLoader(access_token="token", drive_id="drive123", max_files=1)

        root_children = {
            "value": [
                {"id": "a", "name": "a.txt", "file": {"mimeType": "text/plain"}},
                {"id": "b", "name": "b.txt", "file": {"mimeType": "text/plain"}},
            ]
        }

        with (
            patch.object(loader, "_graph_get_json", return_value=root_children),
            patch.object(loader, "_graph_get_bytes", return_value=b"x"),
        ):
            docs = loader.load()

        assert len(docs) == 1
        assert docs[0].metadata["item_id"] == "a"

    def test_aload_runs(self):
        loader = OneDriveLoader(access_token="token", drive_id="drive123")

        with (
            patch.object(
                loader,
                "_graph_get_json",
                return_value={
                    "value": [
                        {"id": "file-1", "name": "note.txt", "file": {"mimeType": "text/plain"}}
                    ]
                },
            ),
            patch.object(loader, "_graph_get_bytes", return_value=b"test"),
        ):
            docs = asyncio.run(loader.aload())

        assert len(docs) == 1
        assert docs[0].text == "test"

    def test_extract_with_supported_loader_returns_none_for_unsupported_ext(self):
        loader = OneDriveLoader(access_token="token", drive_id="drive123")
        out = loader._extract_with_supported_loader("blob.bin", b"raw", ".bin")
        assert out is None

    def test_aload_is_coroutine_function(self):
        import inspect

        assert inspect.iscoroutinefunction(OneDriveLoader(access_token="t", drive_id="d").aload)
