from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


class TestDropboxLoader:
    def _make_loader(self, **kwargs):
        from synapsekit.loaders.dropbox import DropboxLoader

        return DropboxLoader(
            access_token=kwargs.get("access_token", "test-token"),
            path=kwargs.get("path", "/test"),
            limit=kwargs.get("limit"),
        )

    def _make_entry(self, name, path, tag="file", **kwargs):
        return {
            ".tag": tag,
            "name": name,
            "path_display": path,
            "server_modified": kwargs.get("modified", "2024-01-01T00:00:00Z"),
        }

    def _make_list_result(self, entries, has_more=False, cursor="cursor1"):
        result = MagicMock()
        result.entries = entries
        result.has_more = has_more
        result.cursor = cursor
        return result

    def _mock_dropbox(self, mock_dbx):
        """Create mock dropbox module and return it."""
        mock_dropbox = MagicMock()
        mock_dropbox.Dropbox.return_value = mock_dbx
        return mock_dropbox

    def test_import_error_without_dropbox(self):
        loader = self._make_loader()
        with patch.dict("sys.modules", {"dropbox": None}):
            with pytest.raises(ImportError, match="dropbox required"):
                loader.load()

    def test_load_single_text_file(self):
        loader = self._make_loader()
        entry = self._make_entry("test.txt", "/test/test.txt")
        list_result = self._make_list_result([entry])

        mock_dbx = MagicMock()
        mock_dbx.files_list_folder.return_value = list_result
        mock_dbx.files_download_to_file.side_effect = lambda fp, dp: open(fp, "wb").write(
            b"Hello, world!"
        )

        mock_dropbox = self._mock_dropbox(mock_dbx)
        with patch.dict(sys.modules, {"dropbox": mock_dropbox}):
            docs = loader.load()

        assert len(docs) == 1
        assert docs[0].text == "Hello, world!"
        assert docs[0].metadata["source"] == "dropbox"
        assert docs[0].metadata["filename"] == "test.txt"
        assert docs[0].metadata["path"] == "/test/test.txt"
        assert docs[0].metadata["modified"] == "2024-01-01T00:00:00Z"

    def test_load_multiple_files(self):
        loader = self._make_loader()
        entries = [
            self._make_entry("a.txt", "/test/a.txt"),
            self._make_entry("b.md", "/test/b.md"),
            self._make_entry("c.py", "/test/c.py"),
        ]
        list_result = self._make_list_result(entries)

        mock_dbx = MagicMock()
        mock_dbx.files_list_folder.return_value = list_result
        contents = [b"Text A", b"Markdown B", b"Python C"]
        call_count = [0]

        def write_content(fp, dp):
            with open(fp, "wb") as f:
                f.write(contents[call_count[0]])
            call_count[0] += 1

        mock_dbx.files_download_to_file.side_effect = write_content

        mock_dropbox = self._mock_dropbox(mock_dbx)
        with patch.dict(sys.modules, {"dropbox": mock_dropbox}):
            docs = loader.load()

        assert len(docs) == 3
        assert docs[0].text == "Text A"
        assert docs[1].text == "Markdown B"
        assert docs[2].text == "Python C"

    def test_skip_folders(self):
        loader = self._make_loader()
        entries = [
            self._make_entry("file.txt", "/test/file.txt", tag="file"),
            self._make_entry("folder", "/test/folder", tag="folder"),
            self._make_entry("doc.md", "/test/doc.md", tag="file"),
        ]
        list_result = self._make_list_result(entries)

        mock_dbx = MagicMock()
        mock_dbx.files_list_folder.return_value = list_result
        mock_dbx.files_download_to_file.side_effect = lambda fp, dp: open(fp, "wb").write(
            b"content"
        )

        mock_dropbox = self._mock_dropbox(mock_dbx)
        with patch.dict(sys.modules, {"dropbox": mock_dropbox}):
            docs = loader.load()

        assert len(docs) == 2

    def test_skip_unsupported_extensions(self):
        loader = self._make_loader()
        entries = [
            self._make_entry("image.png", "/test/image.png"),
            self._make_entry("video.mp4", "/test/video.mp4"),
            self._make_entry("doc.pdf", "/test/doc.pdf"),
            self._make_entry("text.txt", "/test/text.txt"),
        ]
        list_result = self._make_list_result(entries)

        mock_dbx = MagicMock()
        mock_dbx.files_list_folder.return_value = list_result
        download_count = [0]

        def write_content(fp, dp):
            download_count[0] += 1
            with open(fp, "wb") as f:
                f.write(b"content")

        mock_dbx.files_download_to_file.side_effect = write_content

        mock_dropbox = self._mock_dropbox(mock_dbx)
        with patch.dict(sys.modules, {"dropbox": mock_dropbox}):
            docs = loader.load()

        assert len(docs) == 1
        assert download_count[0] == 1

    def test_pagination_with_cursor(self):
        loader = self._make_loader()
        page1_entries = [
            self._make_entry("file1.txt", "/test/file1.txt"),
            self._make_entry("file2.txt", "/test/file2.txt"),
        ]
        page2_entries = [
            self._make_entry("file3.txt", "/test/file3.txt"),
        ]

        page1_result = self._make_list_result(page1_entries, has_more=True, cursor="cursor1")
        page2_result = self._make_list_result(page2_entries, has_more=False)

        mock_dbx = MagicMock()
        mock_dbx.files_list_folder.return_value = page1_result
        mock_dbx.files_list_folder_continue.return_value = page2_result
        mock_dbx.files_download_to_file.side_effect = lambda fp, dp: open(fp, "wb").write(
            b"content"
        )

        mock_dropbox = self._mock_dropbox(mock_dbx)
        with patch.dict(sys.modules, {"dropbox": mock_dropbox}):
            docs = loader.load()

        assert len(docs) == 3
        mock_dbx.files_list_folder_continue.assert_called_once_with("cursor1")

    def test_limit_stops_early(self):
        loader = self._make_loader(limit=2)
        entries = [
            self._make_entry("file1.txt", "/test/file1.txt"),
            self._make_entry("file2.txt", "/test/file2.txt"),
            self._make_entry("file3.txt", "/test/file3.txt"),
            self._make_entry("file4.txt", "/test/file4.txt"),
        ]
        list_result = self._make_list_result(entries)

        mock_dbx = MagicMock()
        mock_dbx.files_list_folder.return_value = list_result
        download_count = [0]

        def write_content(fp, dp):
            download_count[0] += 1
            with open(fp, "wb") as f:
                f.write(b"content")

        mock_dbx.files_download_to_file.side_effect = write_content

        mock_dropbox = self._mock_dropbox(mock_dbx)
        with patch.dict(sys.modules, {"dropbox": mock_dropbox}):
            docs = loader.load()

        assert len(docs) == 2
        assert download_count[0] == 2

    def test_empty_folder_returns_empty_list(self):
        loader = self._make_loader()
        list_result = self._make_list_result([])

        mock_dbx = MagicMock()
        mock_dbx.files_list_folder.return_value = list_result

        mock_dropbox = self._mock_dropbox(mock_dbx)
        with patch.dict(sys.modules, {"dropbox": mock_dropbox}):
            docs = loader.load()

        assert docs == []

    def test_download_error_skips_file(self):
        loader = self._make_loader()
        entries = [
            self._make_entry("good.txt", "/test/good.txt"),
            self._make_entry("bad.txt", "/test/bad.txt"),
            self._make_entry("also_good.txt", "/test/also_good.txt"),
        ]
        list_result = self._make_list_result(entries)

        mock_dbx = MagicMock()
        mock_dbx.files_list_folder.return_value = list_result
        call_count = [0]

        def write_content(fp, dp):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("Download failed")
            with open(fp, "wb") as f:
                f.write(b"content")

        mock_dbx.files_download_to_file.side_effect = write_content

        mock_dropbox = self._mock_dropbox(mock_dbx)
        with patch.dict(sys.modules, {"dropbox": mock_dropbox}):
            docs = loader.load()

        assert len(docs) == 2

    def test_decode_errors_skips_file(self):
        loader = self._make_loader()
        entries = [
            self._make_entry("good.txt", "/test/good.txt"),
            self._make_entry("bad.txt", "/test/bad.txt"),
        ]
        list_result = self._make_list_result(entries)

        mock_dbx = MagicMock()
        mock_dbx.files_list_folder.return_value = list_result
        call_count = [0]

        def write_content(fp, dp):
            call_count[0] += 1
            if call_count[0] == 2:
                with open(fp, "wb") as f:
                    f.write(b"\x80\x81\x82\x83")
            else:
                with open(fp, "wb") as f:
                    f.write(b"valid content")

        mock_dbx.files_download_to_file.side_effect = write_content

        mock_dropbox = self._mock_dropbox(mock_dbx)
        with patch.dict(sys.modules, {"dropbox": mock_dropbox}):
            docs = loader.load()

        # Both files should be processed (UTF-8 with errors='ignore' won't fail)
        assert len(docs) == 2

    def test_supported_extensions(self):
        loader = self._make_loader()
        extensions = [
            "file.txt",
            "file.md",
            "file.py",
            "file.js",
            "file.ts",
            "file.java",
            "file.json",
            "file.csv",
            "file.xml",
            "file.yaml",
            "file.html",
            "file.css",
            "file.cpp",
            "file.go",
            "file.rs",
        ]
        entries = [self._make_entry(ext, f"/test/{ext}") for ext in extensions]
        list_result = self._make_list_result(entries)

        mock_dbx = MagicMock()
        mock_dbx.files_list_folder.return_value = list_result
        mock_dbx.files_download_to_file.side_effect = lambda fp, dp: open(fp, "wb").write(
            b"content"
        )

        mock_dropbox = self._mock_dropbox(mock_dbx)
        with patch.dict(sys.modules, {"dropbox": mock_dropbox}):
            docs = loader.load()

        assert len(docs) == len(extensions)

    def test_missing_metadata_fields(self):
        loader = self._make_loader()
        # Entry with a valid filename but missing optional fields
        entry = {".tag": "file", "name": "test.txt"}
        list_result = self._make_list_result([entry])

        mock_dbx = MagicMock()
        mock_dbx.files_list_folder.return_value = list_result
        mock_dbx.files_download_to_file.side_effect = lambda fp, dp: open(fp, "wb").write(
            b"content"
        )

        mock_dropbox = self._mock_dropbox(mock_dbx)
        with patch.dict(sys.modules, {"dropbox": mock_dropbox}):
            docs = loader.load()

        assert len(docs) == 1
        assert docs[0].metadata["filename"] == "test.txt"
        assert docs[0].metadata["path"] == ""
        assert docs[0].metadata["modified"] == ""

    def test_list_folder_error_returns_empty(self):
        loader = self._make_loader()

        mock_dbx = MagicMock()
        mock_dbx.files_list_folder.side_effect = Exception("API error")

        mock_dropbox = self._mock_dropbox(mock_dbx)
        with patch.dict(sys.modules, {"dropbox": mock_dropbox}):
            docs = loader.load()

        assert docs == []

    def test_normalise_entry_with_dict_passthrough(self):
        """Dict entries (test mocks) pass through unchanged."""
        from synapsekit.loaders.dropbox import DropboxLoader

        entry = {
            ".tag": "file",
            "name": "a.txt",
            "path_display": "/a.txt",
            "server_modified": "2026-01-01",
        }
        result = DropboxLoader._normalise_entry(entry)
        assert result is entry

    def test_normalise_entry_with_sdk_file_object(self):
        """Real Dropbox SDK FileMetadata objects are normalised correctly."""
        import datetime

        from synapsekit.loaders.dropbox import DropboxLoader

        class FileMetadata:
            name = "report.txt"
            path_display = "/docs/report.txt"
            server_modified = datetime.datetime(2026, 4, 1, 12, 0, tzinfo=datetime.timezone.utc)

        result = DropboxLoader._normalise_entry(FileMetadata())
        assert result[".tag"] == "file"
        assert result["name"] == "report.txt"
        assert result["path_display"] == "/docs/report.txt"
        assert "2026-04-01" in result["server_modified"]

    def test_normalise_entry_with_sdk_folder_object(self):
        """Real Dropbox SDK FolderMetadata objects get tag 'folder'."""
        from synapsekit.loaders.dropbox import DropboxLoader

        class FolderMetadata:
            name = "docs"
            path_display = "/docs"
            server_modified = None

        result = DropboxLoader._normalise_entry(FolderMetadata())
        assert result[".tag"] == "folder"

    @pytest.mark.asyncio
    async def test_aload(self):
        loader = self._make_loader()
        entry = self._make_entry("test.txt", "/test/test.txt")
        list_result = self._make_list_result([entry])

        mock_dbx = MagicMock()
        mock_dbx.files_list_folder.return_value = list_result
        mock_dbx.files_download_to_file.side_effect = lambda fp, dp: open(fp, "wb").write(
            b"Async content"
        )

        mock_dropbox = self._mock_dropbox(mock_dbx)
        with patch.dict(sys.modules, {"dropbox": mock_dropbox}):
            docs = await loader.aload()

        assert len(docs) == 1
        assert docs[0].text == "Async content"
        assert docs[0].metadata["source"] == "dropbox"
