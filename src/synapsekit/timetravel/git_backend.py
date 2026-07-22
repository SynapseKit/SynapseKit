"""Git backend and point-in-time scoping for Time-Travel Codebase."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .agent import TimeTravelAgent


def _parse_datetime(val: str | datetime | None) -> datetime | None:
    if val is None:
        return None
    if isinstance(val, datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=UTC)
        return val
    try:
        dt = datetime.fromisoformat(val)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt
    except ValueError:
        # Fallback for simple date YYYY-MM-DD
        try:
            parts = [int(p) for p in val.split("-")]
            if len(parts) == 3:
                return datetime(parts[0], parts[1], parts[2], tzinfo=UTC)
        except Exception:
            pass
        return None


@dataclass(frozen=True)
class CommitInfo:
    """Represents metadata for a single git commit."""

    hash: str
    author: str
    date: datetime
    subject: str
    body: str = ""
    files_changed: list[str] = field(default_factory=list)


class GitBackend:
    """Wrapper around git CLI commands for code history access."""

    def __init__(self, repo_path: str | Path = ".") -> None:
        self.repo_path = Path(repo_path).resolve()

    def _run_git(self, args: list[str]) -> str:
        """Run a git command in the repository directory."""
        cmd = ["git", *args]
        try:
            res = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=True,
            )
            return res.stdout
        except (subprocess.CalledProcessError, FileNotFoundError) as err:
            raise RuntimeError(f"Git command failed: {' '.join(cmd)} ({err})") from err

    def log(
        self,
        path: str | Path | None = None,
        follow: bool = True,
        since: str | datetime | None = None,
        until: str | datetime | None = None,
        max_count: int | None = None,
    ) -> list[CommitInfo]:
        """Fetch commit history for the repo or a specific file path."""
        delimiter = "---COMMIT_DELIMITER---"
        field_delimiter = "---FIELD_DELIMITER---"
        format_str = f"{delimiter}%n%H{field_delimiter}%an{field_delimiter}%aI{field_delimiter}%s{field_delimiter}%b"

        args = ["log", f"--format={format_str}", "--name-only"]
        if follow and path:
            args.append("--follow")
        if max_count:
            args.extend(["-n", str(max_count)])

        since_dt = _parse_datetime(since)
        if since_dt:
            args.append(f"--since={since_dt.isoformat()}")

        until_dt = _parse_datetime(until)
        if until_dt:
            args.append(f"--until={until_dt.isoformat()}")

        if path:
            args.extend(["--", str(path)])

        try:
            output = self._run_git(args)
        except RuntimeError:
            return []

        commits: list[CommitInfo] = []
        raw_blocks = output.split(delimiter)
        for block in raw_blocks:
            block = block.strip()
            if not block:
                continue

            lines = block.split("\n")
            header_line = lines[0]
            parts = header_line.split(field_delimiter)
            if len(parts) < 4:
                continue

            commit_hash = parts[0].strip()
            author = parts[1].strip()
            date_str = parts[2].strip()
            subject = parts[3].strip()
            body = parts[4].strip() if len(parts) > 4 else ""

            files_changed: list[str] = []
            for line in lines[1:]:
                clean_line = line.strip()
                if clean_line and not clean_line.startswith("---FIELD_DELIMITER---"):
                    files_changed.append(clean_line)

            try:
                dt = datetime.fromisoformat(date_str)
            except ValueError:
                dt = datetime.now(UTC)

            commits.append(
                CommitInfo(
                    hash=commit_hash,
                    author=author,
                    date=dt,
                    subject=subject,
                    body=body,
                    files_changed=files_changed,
                )
            )

        return commits

    def diff(
        self,
        commit_a: str,
        commit_b: str = "HEAD",
        path: str | Path | None = None,
    ) -> str:
        """Get unified diff between two commits."""
        args = ["diff", commit_a, commit_b]
        if path:
            args.extend(["--", str(path)])
        try:
            return self._run_git(args)
        except RuntimeError:
            return ""

    def show(self, commit: str, path: str | Path) -> str:
        """Get file content at a specific commit."""
        rel_path = str(path).replace("\\", "/")
        try:
            return self._run_git(["show", f"{commit}:{rel_path}"])
        except RuntimeError:
            return ""

    def file_at(self, commit: str, path: str | Path) -> str:
        """Alias for show()."""
        return self.show(commit, path)

    def find_commit_at(self, date: str | datetime) -> str:
        """Find the commit hash closest to (or before/at) a given date."""
        target_dt = _parse_datetime(date)
        if target_dt is None:
            return "HEAD"

        iso_str = target_dt.isoformat()
        try:
            output = self._run_git(["log", "-1", f"--until={iso_str}", "--format=%H"])
            commit_hash = output.strip()
            if commit_hash:
                return commit_hash
        except RuntimeError:
            pass

        try:
            output = self._run_git(["log", "-1", "--reverse", "--format=%H"])
            return output.strip() or "HEAD"
        except RuntimeError:
            return "HEAD"

    def blame(self, path: str | Path, commit: str | None = None) -> list[dict[str, Any]]:
        """Run git blame on a file and return structured line ownership."""
        args = ["blame", "--line-porcelain"]
        if commit:
            args.append(commit)
        args.extend(["--", str(path)])

        try:
            output = self._run_git(args)
        except RuntimeError:
            return []

        results: list[dict[str, Any]] = []
        current: dict[str, Any] = {}
        for line in output.splitlines():
            if line.startswith("\t"):
                current["content"] = line[1:]
                results.append(current)
                current = {}
            else:
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    key, val = parts[0], parts[1]
                    if key in ("author", "author-mail", "author-time", "summary"):
                        current[key] = val
                    elif len(parts[0]) == 40:
                        current["commit"] = parts[0]
        return results

    def list_files(self, commit: str | None = None) -> list[str]:
        """List tracked files at a commit or HEAD."""
        if commit:
            args = ["ls-tree", "-r", "--name-only", commit]
        else:
            args = ["ls-files"]
        try:
            output = self._run_git(args)
            return [f.strip() for f in output.splitlines() if f.strip()]
        except RuntimeError:
            return []


@dataclass
class AsOf:
    """Context wrapper for point-in-time (as-of date) queries."""

    agent: TimeTravelAgent
    date: datetime
    commit: str

    async def query(self, question: str) -> str:
        """Query codebase state as of the configured date."""
        return await self.agent._query_as_of(question, self.date, self.commit)

    async def detect_drift(self, symbol: str) -> Any:
        """Detect drift relative to the configured date context."""
        return await self.agent.detect_drift(symbol, as_of_date=self.date)

    async def timeline(self, file_or_symbol: str) -> list[Any]:
        """Get timeline up to the configured date."""
        return await self.agent.timeline(file_or_symbol, until=self.date)
