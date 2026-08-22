"""Evolution index for tracking file and symbol-level changes over time."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from .git_backend import CommitInfo, GitBackend, _parse_datetime


@dataclass(frozen=True)
class EvolutionEntry:
    """Represents a single change event for a file or code symbol."""

    file_path: str
    symbol: str | None
    commit: CommitInfo
    diff_snippet: str
    change_type: Literal["added", "modified", "deleted", "renamed"]
    lines_added: int
    lines_removed: int
    pr_number: int | None = None


class EvolutionIndex:
    """Index that parses git commit history into file and symbol evolution entries."""

    def __init__(self, backend: GitBackend) -> None:
        self.backend = backend
        self._entries: list[EvolutionEntry] = []

    @staticmethod
    def _extract_pr_number(text: str) -> int | None:
        """Extract PR number from commit subject or body (e.g. #123)."""
        match = re.search(r"#(\d+)", text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        return None

    @staticmethod
    def _extract_python_symbols(source_code: str) -> set[str]:
        """Extract class and top-level function names from Python source code."""
        if not source_code.strip():
            return set()
        try:
            tree = ast.parse(source_code)
            symbols = set()
            for node in tree.body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.add(node.name)
            return symbols
        except Exception:
            return set()

    def build(
        self,
        paths: list[str] | None = None,
        since: str | datetime | None = None,
        until: str | datetime | None = None,
        max_count: int | None = None,
    ) -> list[EvolutionEntry]:
        """Build evolution entries by traversing commit history."""
        target_paths: list[str | None] = list(paths) if paths else [None]
        all_entries: list[EvolutionEntry] = []

        for p in target_paths:
            commits = self.backend.log(
                path=p, follow=True, since=since, until=until, max_count=max_count
            )
            for i, commit in enumerate(commits):
                prev_commit = commits[i + 1].hash if i + 1 < len(commits) else f"{commit.hash}~1"
                diff_text = self.backend.diff(prev_commit, commit.hash, path=p)

                pr_num = self._extract_pr_number(commit.subject) or self._extract_pr_number(
                    commit.body
                )

                files_to_check = commit.files_changed or ([p] if p else [])
                for filepath in files_to_check:
                    if not filepath:
                        continue

                    # Calculate lines added/removed
                    added = 0
                    removed = 0
                    snippet_lines: list[str] = []
                    for line in diff_text.splitlines():
                        if line.startswith("+") and not line.startswith("+++"):
                            added += 1
                            if len(snippet_lines) < 20:
                                snippet_lines.append(line)
                        elif line.startswith("-") and not line.startswith("---"):
                            removed += 1
                            if len(snippet_lines) < 20:
                                snippet_lines.append(line)

                    snippet = "\n".join(snippet_lines[:15])

                    change_type: Literal["added", "modified", "deleted", "renamed"] = "modified"
                    if prev_commit.endswith("~1") and i == len(commits) - 1:
                        change_type = "added"
                    elif removed > 0 and added == 0:
                        change_type = "deleted"
                    elif added > 0 and removed == 0 and prev_commit == f"{commit.hash}~1":
                        change_type = "added"

                    # Determine symbols if python file
                    symbols: set[str] = set()
                    if filepath.endswith(".py"):
                        curr_content = self.backend.file_at(commit.hash, filepath)
                        symbols = self._extract_python_symbols(curr_content)

                    if symbols:
                        for sym in sorted(symbols):
                            entry = EvolutionEntry(
                                file_path=filepath,
                                symbol=sym,
                                commit=commit,
                                diff_snippet=snippet,
                                change_type=change_type,
                                lines_added=added,
                                lines_removed=removed,
                                pr_number=pr_num,
                            )
                            all_entries.append(entry)
                    else:
                        entry = EvolutionEntry(
                            file_path=filepath,
                            symbol=None,
                            commit=commit,
                            diff_snippet=snippet,
                            change_type=change_type,
                            lines_added=added,
                            lines_removed=removed,
                            pr_number=pr_num,
                        )
                        all_entries.append(entry)

        # Deduplicate entries by (commit.hash, file_path, symbol)
        seen = set()
        deduped = []
        for entry in all_entries:
            key = (entry.commit.hash, entry.file_path, entry.symbol)
            if key not in seen:
                seen.add(key)
                deduped.append(entry)

        self._entries = deduped
        return deduped

    def ensure_built(self) -> list[EvolutionEntry]:
        """Build entries if not already built, otherwise return the cached entries.

        Lets callers share one `EvolutionIndex` across multiple consumers
        without repeating the underlying `git log`/`git diff`/AST-parse walk.
        """
        if not self._entries:
            self.build()
        return self._entries

    def query(
        self,
        file_or_symbol: str,
        since: str | datetime | None = None,
        until: str | datetime | None = None,
    ) -> list[EvolutionEntry]:
        """Query evolution entries matching a file path or symbol name."""
        if not self._entries:
            self.build()

        since_dt = _parse_datetime(since)
        until_dt = _parse_datetime(until)

        results = []
        target = file_or_symbol.strip()
        for entry in self._entries:
            if since_dt and entry.commit.date < since_dt:
                continue
            if until_dt and entry.commit.date > until_dt:
                continue

            matches_file = target in entry.file_path or entry.file_path.endswith(target)
            matches_symbol = entry.symbol and (target == entry.symbol or target in entry.symbol)

            if matches_file or matches_symbol:
                results.append(entry)

        return results

    def timeline(
        self,
        file_or_symbol: str,
        since: str | datetime | None = None,
        until: str | datetime | None = None,
    ) -> list[EvolutionEntry]:
        """Return chronological timeline of evolution entries (oldest to newest)."""
        entries = self.query(file_or_symbol, since=since, until=until)
        return sorted(entries, key=lambda e: e.commit.date)
