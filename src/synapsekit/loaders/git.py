from __future__ import annotations

import tempfile
from pathlib import PurePosixPath

from .base import Document


class GitLoader:
    """Load files from a Git repository at a specific revision."""

    def __init__(
        self,
        repo: str,
        revision: str = "HEAD",
        glob_pattern: str = "**/*",
    ) -> None:
        self._repo = repo
        self._revision = revision
        self._glob_pattern = glob_pattern

    def _is_remote_repo(self, repo: str) -> bool:
        return repo.startswith(("http://", "https://", "git@", "ssh://", "file://"))

    def _iter_matching_blobs(self, tree):
        for item in tree.traverse():
            if getattr(item, "type", None) != "blob":
                continue
            if PurePosixPath(item.path).match(self._glob_pattern):
                yield item

    def _load_from_repo(self, repo) -> list[Document]:
        commit = repo.commit(self._revision)
        docs: list[Document] = []

        for blob in sorted(self._iter_matching_blobs(commit.tree), key=lambda b: b.path):
            text = blob.data_stream.read().decode("utf-8", errors="replace")
            metadata = {
                "path": blob.path,
                "commit_hash": commit.hexsha,
                "author": str(commit.author),
                "date": commit.committed_datetime.isoformat(),
            }
            docs.append(Document(text=text, metadata=metadata))

        return docs

    def load(self) -> list[Document]:
        try:
            import git
        except ImportError:
            raise ImportError("gitpython required: pip install synapsekit[git]") from None

        if self._is_remote_repo(self._repo):
            with tempfile.TemporaryDirectory(prefix="synapsekit-git-") as tmpdir:
                repo = git.Repo.clone_from(self._repo, tmpdir)
                try:
                    return self._load_from_repo(repo)
                finally:
                    if hasattr(repo, "close"):
                        repo.close()

        repo = git.Repo(self._repo)
        try:
            return self._load_from_repo(repo)
        finally:
            if hasattr(repo, "close"):
                repo.close()
