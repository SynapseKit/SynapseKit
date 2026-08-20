"""TimelineReconstructor — merge multi-source events into a single timeline."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path

from .types import Citation, TimelineEvent

logger = logging.getLogger(__name__)


class TimelineReconstructor:
    """Collects events from git, Slack, email, and markdown, then merges chronologically."""

    def __init__(self, repo_path: str | Path = ".") -> None:
        self.repo_path = Path(repo_path).resolve()

    async def reconstruct(
        self,
        query: str,
        *,
        include_git: bool = True,
        slack_bot_token: str | None = None,
        slack_channel_ids: list[str] | None = None,
        email_imap_server: str | None = None,
        email_address: str | None = None,
        email_password: str | None = None,
        email_folder: str = "INBOX",
        markdown_roots: list[str | Path] | None = None,
        max_events: int = 200,
    ) -> list[TimelineEvent]:
        """Gather and merge events from all configured sources."""
        tasks: list[asyncio.Task[list[TimelineEvent]]] = []

        if include_git:
            tasks.append(asyncio.create_task(self._git_events(query, max_events)))

        if slack_bot_token and slack_channel_ids:
            tasks.append(
                asyncio.create_task(
                    self._slack_events(query, slack_bot_token, slack_channel_ids)
                )
            )

        if email_imap_server and email_address and email_password:
            tasks.append(
                asyncio.create_task(
                    self._email_events(
                        query,
                        email_imap_server,
                        email_address,
                        email_password,
                        email_folder,
                    )
                )
            )

        if markdown_roots:
            tasks.append(
                asyncio.create_task(
                    self._markdown_events(query, [Path(r) for r in markdown_roots])
                )
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)
        events: list[TimelineEvent] = []
        for result in results:
            if isinstance(result, BaseException):
                logger.warning("Source failed in TimelineReconstructor: %s", result)
                continue
            events.extend(result)

        events.sort(key=lambda e: e.timestamp)
        return events[:max_events]

    async def _git_events(self, query: str, max_events: int) -> list[TimelineEvent]:
        """Extract timeline events from git history."""
        from ..timetravel.evolution_index import EvolutionIndex
        from ..timetravel.git_backend import GitBackend

        backend = GitBackend(self.repo_path)
        index = EvolutionIndex(backend)

        entries = await asyncio.to_thread(index.build)
        terms = [t.strip("?,.'\"`") for t in query.split() if len(t) > 2]

        matching = [
            e
            for e in entries
            if any(
                term.lower()
                in (e.file_path + " " + (e.symbol or "") + " " + e.commit.subject).lower()
                for term in terms
            )
        ] or entries[:max_events]

        events: list[TimelineEvent] = []
        for entry in matching[:max_events]:
            pr_ref = f" (PR #{entry.pr_number})" if entry.pr_number else ""
            citation = Citation(
                source_type="git",
                reference=f"commit {entry.commit.hash[:8]}{pr_ref}",
                content_preview=entry.commit.subject,
                timestamp=entry.commit.date,
                metadata={
                    "file_path": entry.file_path,
                    "symbol": entry.symbol,
                    "author": entry.commit.author,
                },
            )
            events.append(
                TimelineEvent(
                    timestamp=entry.commit.date,
                    source_type="git",
                    summary=(
                        f"{entry.change_type} {entry.file_path}"
                        + (f" [{entry.symbol}]" if entry.symbol else "")
                        + f": {entry.commit.subject}"
                    ),
                    citations=[citation],
                    metadata={"commit_hash": entry.commit.hash},
                )
            )

        return events

    async def _slack_events(
        self,
        query: str,
        bot_token: str,
        channel_ids: list[str],
    ) -> list[TimelineEvent]:
        """Extract timeline events from Slack messages."""
        from ..loaders.slack import SlackLoader

        events: list[TimelineEvent] = []
        terms = [t.lower().strip("?,.'\"`") for t in query.split() if len(t) > 2]

        for channel_id in channel_ids:
            loader = SlackLoader(bot_token=bot_token, channel_id=channel_id)
            docs = await loader.aload()
            for doc in docs:
                text_lower = doc.text.lower()
                if not any(term in text_lower for term in terms):
                    continue
                ts_str = str(doc.metadata.get("timestamp", ""))
                try:
                    ts = datetime.fromtimestamp(float(ts_str), tz=UTC)
                except (ValueError, TypeError, OSError):
                    ts = datetime.now(UTC)
                citation = Citation(
                    source_type="slack",
                    reference=f"slack://{channel_id}/{ts_str}",
                    content_preview=doc.text[:200],
                    timestamp=ts,
                    metadata=doc.metadata,
                )
                events.append(
                    TimelineEvent(
                        timestamp=ts,
                        source_type="slack",
                        summary=doc.text[:200],
                        citations=[citation],
                    )
                )

        return events

    async def _email_events(
        self,
        query: str,
        imap_server: str,
        email_address: str,
        password: str,
        folder: str,
    ) -> list[TimelineEvent]:
        """Extract timeline events from email archives."""
        from ..loaders.email import EmailLoader

        loader = EmailLoader(
            imap_server=imap_server,
            email_address=email_address,
            password=password,
            folder=folder,
        )
        docs = await loader.aload()
        terms = [t.lower().strip("?,.'\"`") for t in query.split() if len(t) > 2]
        events: list[TimelineEvent] = []

        for doc in docs:
            text_lower = (doc.text + " " + doc.metadata.get("subject", "")).lower()
            if not any(term in text_lower for term in terms):
                continue
            date_str = doc.metadata.get("date", "")
            try:
                ts = datetime.fromisoformat(date_str) if date_str else datetime.now(UTC)
            except ValueError:
                ts = datetime.now(UTC)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            citation = Citation(
                source_type="email",
                reference=f"email://{doc.metadata.get('email_id', 'unknown')}",
                content_preview=doc.text[:200],
                timestamp=ts,
                metadata=doc.metadata,
            )
            events.append(
                TimelineEvent(
                    timestamp=ts,
                    source_type="email",
                    summary=f"{doc.metadata.get('subject', 'No subject')}: {doc.text[:100]}",
                    citations=[citation],
                )
            )

        return events

    async def _markdown_events(
        self,
        query: str,
        roots: list[Path],
    ) -> list[TimelineEvent]:
        """Extract timeline events from local markdown notes."""
        terms = [t.lower().strip("?,.'\"`") for t in query.split() if len(t) > 2]
        events: list[TimelineEvent] = []

        for root in roots:
            root = root.resolve()
            if not root.exists():
                continue
            md_files = list(root.rglob("*.md"))
            for md_file in md_files:
                try:
                    content = await asyncio.to_thread(
                        md_file.read_text, encoding="utf-8"
                    )
                except Exception:
                    continue
                if not any(term in content.lower() for term in terms):
                    continue
                stat = md_file.stat()
                ts = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
                citation = Citation(
                    source_type="markdown",
                    reference=f"file://{md_file}",
                    content_preview=content[:200],
                    timestamp=ts,
                    metadata={"path": str(md_file)},
                )
                events.append(
                    TimelineEvent(
                        timestamp=ts,
                        source_type="markdown",
                        summary=f"Note: {md_file.name} — {content[:100]}",
                        citations=[citation],
                    )
                )

        return events
