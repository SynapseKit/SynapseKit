"""File-level diff engine for Living Memory patches."""

from __future__ import annotations

import difflib
from pathlib import Path


class DiffConflictError(Exception):
    """Raised when a patch cannot be applied due to file changes since proposal."""

    def __init__(self, file_path: str, expected_hash: str, actual_hash: str) -> None:
        self.file_path = file_path
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash
        super().__init__(
            f"Conflict in {file_path}: expected content hash "
            f"{expected_hash[:12]}… but found {actual_hash[:12]}…"
        )


class FileDiffEngine:
    """Generate and apply unified diffs against memory files.

    Provides utilities for creating unified diffs, validating that a
    patch is still applicable (the file hasn't diverged), applying
    patches, and reverting them.
    """

    @staticmethod
    def generate_unified_diff(
        before: str,
        after: str,
        file_path: str = "memory.md",
        *,
        context_lines: int = 3,
    ) -> str:
        """Create a unified diff string between *before* and *after* content."""
        before_lines = before.splitlines(keepends=True)
        after_lines = after.splitlines(keepends=True)

        # Ensure trailing newlines for clean diff output
        if before_lines and not before_lines[-1].endswith("\n"):
            before_lines[-1] += "\n"
        if after_lines and not after_lines[-1].endswith("\n"):
            after_lines[-1] += "\n"

        diff_lines = difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            n=context_lines,
        )
        return "".join(diff_lines)

    @staticmethod
    def compute_similarity(before: str, after: str) -> float:
        """Return a 0.0-1.0 similarity ratio between two strings."""
        matcher = difflib.SequenceMatcher(None, before, after)
        return matcher.ratio()

    @staticmethod
    def validate_patch_applicable(
        file_path: str | Path,
        expected_before: str,
    ) -> tuple[bool, str]:
        """Check whether the file's current content matches the expected snapshot.

        Returns a ``(is_applicable, reason)`` tuple.  A patch is considered
        applicable when the current file content exactly matches, matches
        after whitespace normalization, or is >95% similar.
        """
        path = Path(file_path)
        if not path.exists():
            return False, f"File does not exist: {path}"

        current = path.read_text(encoding="utf-8")
        if current == expected_before:
            return True, "content matches"

        # Allow minor whitespace drift
        if current.strip() == expected_before.strip():
            return True, "content matches (whitespace-normalized)"

        similarity = FileDiffEngine.compute_similarity(current, expected_before)
        if similarity > 0.95:
            return True, f"content closely matches (similarity={similarity:.3f})"

        return False, (
            f"content has diverged (similarity={similarity:.3f}); "
            "file may have been edited since the patch was proposed"
        )

    @staticmethod
    def apply_patch(file_path: str | Path, after_content: str) -> str:
        """Write the *after* content to the file.  Returns the previous content."""
        path = Path(file_path)
        previous = ""
        if path.exists():
            previous = path.read_text(encoding="utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(after_content, encoding="utf-8")
        return previous

    @staticmethod
    def revert_to_content(file_path: str | Path, before_content: str) -> str:
        """Revert a file to the given content snapshot.  Returns the reverted-from content."""
        path = Path(file_path)
        current = ""
        if path.exists():
            current = path.read_text(encoding="utf-8")
        path.write_text(before_content, encoding="utf-8")
        return current

    @staticmethod
    def count_changed_lines(unified_diff: str) -> dict[str, int]:
        """Parse a unified diff and count additions, deletions, and net change."""
        added = 0
        removed = 0
        for line in unified_diff.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                added += 1
            elif line.startswith("-") and not line.startswith("---"):
                removed += 1
        return {"added": added, "removed": removed, "net": added - removed}
