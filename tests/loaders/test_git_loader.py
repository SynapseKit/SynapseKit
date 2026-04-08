from __future__ import annotations

from unittest.mock import patch

import pytest


class TestGitLoader:
    def test_import_error_without_gitpython(self):
        from synapsekit.loaders.git import GitLoader

        with patch.dict("sys.modules", {"git": None}):
            loader = GitLoader("/tmp/repo")
            with pytest.raises(ImportError, match="gitpython required"):
                loader.load()

    def test_load_from_local_repo_revision_and_glob(self, tmp_path):
        git = pytest.importorskip("git")
        from synapsekit.loaders.git import GitLoader

        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        repo = git.Repo.init(repo_dir)

        file_a = repo_dir / "a.txt"
        file_b = repo_dir / "b.md"
        file_a.write_text("v1")
        file_b.write_text("markdown")
        repo.index.add(["a.txt", "b.md"])
        commit1 = repo.index.commit("initial")

        file_a.write_text("v2")
        repo.index.add(["a.txt"])
        repo.index.commit("update a")

        docs = GitLoader(str(repo_dir), revision=commit1.hexsha, glob_pattern="*.txt").load()

        assert len(docs) == 1
        assert docs[0].text == "v1"
        assert docs[0].metadata["path"] == "a.txt"
        assert docs[0].metadata["commit_hash"] == commit1.hexsha
        assert docs[0].metadata["author"]
        assert docs[0].metadata["date"]

    def test_load_from_file_url(self, tmp_path):
        git = pytest.importorskip("git")
        from synapsekit.loaders.git import GitLoader

        remote_repo_dir = tmp_path / "remote_repo"
        remote_repo_dir.mkdir()
        repo = git.Repo.init(remote_repo_dir)

        (remote_repo_dir / "note.txt").write_text("from remote")
        repo.index.add(["note.txt"])
        repo.index.commit("add note")

        file_url = f"file://{remote_repo_dir.as_posix()}"
        docs = GitLoader(file_url, revision="HEAD", glob_pattern="*.txt").load()

        assert len(docs) == 1
        assert docs[0].text == "from remote"
        assert docs[0].metadata["path"] == "note.txt"
